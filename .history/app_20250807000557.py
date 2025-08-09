import os
import json
import base64
import tempfile
import io
import re
from typing import Any, Dict, List
import asyncio
import aiofiles
import httpx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import duckdb
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
AIPIPE_TOKEN = os.getenv('AIPIPE_TOKEN')
AIPipe_BASE_URL = "https://aipipe.org/openrouter/v1"
AIPipe_MODEL = os.getenv('AIPipe_MODEL', 'openai/gpt-4o-mini')

if not AIPIPE_TOKEN:
    logging.warning("AIPIPE_TOKEN environment variable not set. AI calls will fail.")

# --- LLM Client for AI Pipe ---
class AIPipeClient:
    def __init__(self, token: str, base_url: str, model: str):
        self.token = token
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=180.0)
        logging.info(f"Initialized AI Pipe client for model: {model}")

    async def generate_json_response(self, prompt: str) -> Dict[str, Any]:
        if not self.token:
            raise ValueError("AI Pipe token is missing.")
        
        messages = [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": messages, "response_format": {"type": "json_object"}}
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        
        try:
            response = await self.client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            return json.loads(content)
        except (httpx.RequestError, json.JSONDecodeError, KeyError, TypeError) as e:
            logging.error(f"Error generating response from AI Pipe: {e}", exc_info=True)
            raise

# --- Core Data Analyst Agent ---
class DataAnalystAgent:
    def __init__(self, llm_client: AIPipeClient):
        self.llm = llm_client
        # --- FIX: Added the missing temp_dir initialization ---
        self.temp_dir = tempfile.mkdtemp()

    async def _create_plan(self, questions: str, context: Dict) -> Dict:
        prompt = f"""
        You are a data analyst agent. Create a step-by-step JSON execution plan to answer the user's questions.

        **User's Questions:**
        {questions}
        
        **Available Context & Files:**
        {json.dumps(context, indent=2)}

        **Your Task:**
        Create a JSON object with a key "plan".
        If data is scraped or uploaded, it will be available as a DuckDB table named 'data_table'. All SQL steps MUST query this table.
        When creating a plan, use generic column names like 'Rank', 'Year', 'Gross', 'Peak', 'Title'. The execution engine will map them correctly.

        **Supported Actions & Params:**
        - "scrape_website": {{"url": "URL to scrape"}}
        - "run_duckdb_sql": {{"query": "SQL query to run on the 'data_table'."}}
        - "generate_plot": {{"plot_type": "scatterplot", "x_col": "x-axis column", "y_col": "y-axis column", "options": {{"color": "red", "style": "dotted"}}}}
        """
        logging.info("Generating analysis plan...")
        plan = await self.llm.generate_json_response(prompt)
        logging.info(f"Generated Plan: {plan}")
        return plan

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and prepares a dataframe for analysis."""
        df.columns = [re.sub(r'\[.*?\]', '', col).strip() for col in df.columns]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                cleaned_series = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                df[col] = pd.to_numeric(cleaned_series, errors='coerce')
        return df

    def _get_actual_column_name(self, df_columns: List[str], requested_name: str) -> str:
        """Finds the actual column name that best matches the requested name."""
        requested_lower = requested_name.lower()
        for col in df_columns:
            if requested_lower == col.lower():
                return f'"{col}"'
        for col in df_columns:
            if requested_lower in col.lower():
                return f'"{col}"'
        raise ValueError(f"Could not find a column that matches '{requested_name}'. Available columns: {df_columns}")


    async def _execute_plan(self, plan: Dict, uploaded_files: Dict) -> Dict:
        execution_results = {}
        # --- FIX: Establish a single, persistent in-memory database connection ---
        with duckdb.connect(database=':memory:', read_only=False) as con:
            con.execute("INSTALL httpfs; LOAD httpfs;")
            
            main_df = None

            # --- FIX: Logic to load data ONCE at the beginning ---
            is_scrape_in_plan = any(s['action'] == "scrape_website" for s in plan.get("plan", []))
            if is_scrape_in_plan:
                # Find the scrape step and execute it first
                scrape_step = next((s for s in plan['plan'] if s['action'] == 'scrape_website'), None)
                if scrape_step:
                    logging.info(f"Executing pre-emptive scrape step: {scrape_step}")
                    proxy_url = f"https://aipipe.org/proxy/{scrape_step['params']['url']}"
                    async with httpx.AsyncClient() as client:
                        response = await client.get(proxy_url, follow_redirects=True, timeout=60.0)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    table = soup.find('table', {'class': 'wikitable'})
                    if table:
                        df = pd.read_html(io.StringIO(str(table)))[0]
                        main_df = self._clean_dataframe(df)
                        con.register('data_table', main_df)
                        execution_results["step_0_scrape_website"] = f"Scraped data. Columns: {main_df.columns.tolist()}"
            
            elif uploaded_files:
                 # If no scraping, load the first uploaded CSV
                 first_file_name = next(iter(uploaded_files))
                 main_df = pd.read_csv(uploaded_files[first_file_name])
                 main_df = self._clean_dataframe(main_df)
                 con.register('data_table', main_df)
                 logging.info(f"Loaded {first_file_name} into 'data_table'.")

            # --- Now execute the rest of the plan ---
            for i, step in enumerate(plan.get("plan", [])):
                action, params = step.get("action"), step.get("params", {})
                step_key = f"step_{i+1}_{action}"
                
                # Skip the scrape step as it's already done
                if action == "scrape_website":
                    continue

                logging.info(f"Executing step {i+1}: {action} with params {params}")
                
                try:
                    if action == "run_duckdb_sql":
                        if main_df is None: raise ValueError("No data available to query.")
                        
                        query = params["query"]
                        potential_cols = re.findall(r'\b([A-Za-z_]+)\b', query)
                        
                        for col_name in set(potential_cols):
                            try:
                                actual_name = self._get_actual_column_name(main_df.columns, col_name)
                                query = re.sub(r'\b' + re.escape(col_name) + r'\b', actual_name, query, flags=re.IGNORECASE)
                            except ValueError: pass 
                        
                        logging.info(f"Executing rewritten SQL: {query}")
                        result_df = con.execute(query).df()
                        execution_results[step_key] = result_df.to_dict(orient='records')

                    elif action == "generate_plot":
                        if main_df is None: raise ValueError("No data available to plot.")
                        
                        df = main_df
                        fig, ax = plt.subplots(figsize=(8, 5))
                        
                        x_col_actual = self._get_actual_column_name(df.columns, params["x_col"]).strip('"')
                        y_col_actual = self._get_actual_column_name(df.columns, params["y_col"]).strip('"')
                        
                        valid_data = df[[x_col_actual, y_col_actual]].dropna()

                        ax.scatter(valid_data[x_col_actual], valid_data[y_col_actual], alpha=0.6)
                        
                        if "options" in params and not valid_data.empty:
                            z = np.polyfit(valid_data[x_col_actual], valid_data[y_col_actual], 1)
                            p = np.poly1d(z)
                            ax.plot(valid_data[x_col_actual], p(valid_data[x_col_actual]), color=params["options"].get("color", "red"), linestyle=params["options"].get("style", "dotted"))
                        
                        ax.set_xlabel(x_col_actual); ax.set_ylabel(y_col_actual); plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=90)
                        plt.close(fig)
                        base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                        
                        if len(base64_img) > 100000:
                             execution_results[step_key] = "Error: Generated image is too large."
                        else:
                            execution_results[step_key] = f"data:image/png;base64,{base64_img}"
                    
                except Exception as e:
                    logging.error(f"Error executing step {i+1} ({action}): {e}", exc_info=True)
                    execution_results[step_key] = {"error": str(e)}
        return execution_results

    async def _synthesize_answer(self, questions: str, results: Dict) -> Any:
        prompt = f"""
        You are a synthesizer agent. Construct the final answer based on the user's request and the execution results.
        You MUST format your response exactly as requested (e.g., a JSON array or object).
        Use the 'Execution Results' to find the values. If a step has an error, state that in a user-friendly way.

        **Original Questions:**
        {questions}

        **Execution Results:**
        {json.dumps(results, indent=2)}

        Provide only the final, formatted JSON output. Do not include explanations.
        """
        logging.info("Synthesizing final answer...")
        return await self.llm.generate_json_response(prompt)

    async def analyze(self, questions: str, files: List[UploadFile]) -> Any:
        temp_files = {}
        context = {"files": []}
        for file in files:
            if file.filename != "questions.txt":
                file_path = os.path.join(self.temp_dir, file.filename)
                async with aiofiles.open(file_path, 'wb') as f: await f.write(await file.read())
                temp_files[file.filename] = file_path
                context["files"].append(file.filename)
        
        urls = re.findall(r'http[s]?://\S+', questions)
        if urls: context["urls_in_question"] = urls
            
        plan = await self._create_plan(questions, context)
        results = await self._execute_plan(plan, temp_files)
        return await self._synthesize_answer(questions, results)

# --- FastAPI App ---
app = FastAPI(title="AI Pipe Data Analyst Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

aipipe_client = AIPipeClient(AIPIPE_TOKEN, AIPipe_BASE_URL, AIPipe_MODEL)
agent = DataAnalystAgent(aipipe_client)

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    questions_file = next((f for f in files if f.filename == "questions.txt"), None)
    if not questions_file: raise HTTPException(status_code=400, detail="questions.txt is required.")
    
    questions_content = (await questions_file.read()).decode('utf-8')
    
    try:
        result = await agent.analyze(questions_content, files)
        return JSONResponse(content=result)
    except Exception as e:
        logging.error(f"Unhandled error in analysis endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": f"AI Pipe Data Analyst Agent is running with model {AIPipe_MODEL}."}
def create_test_files():
    """Creates the test files needed for the analysis."""
    logging.info("Creating test files...")
