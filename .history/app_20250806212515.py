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
        self.temp_dir = tempfile.mkdtemp()
        self.file_paths = {}

    async def _create_plan(self, questions: str, context: Dict) -> Dict:
        prompt = f"""
        You are a data analyst agent. Create a step-by-step JSON execution plan to answer the user's questions.

        **User's Questions:**
        {questions}
        
        **Available Context & Files:**
        {json.dumps(context, indent=2)}

        **Your Task:**
        Create a JSON object with a key "plan". This key holds a list of action steps.
        If data is scraped, it will be saved as "scraped_data.csv". Subsequent steps MUST use this filename.
        When creating a plan, use generic column names like 'Rank', 'Year', 'Gross'. The execution engine will map them to the correct columns.

        **Supported Actions & Params:**
        - "scrape_website": {{"url": "URL to scrape"}}
        - "run_duckdb_sql": {{"query": "SQL query to run on the scraped or uploaded data."}}
        - "generate_plot": {{"file_name": "CSV file name", "plot_type": "scatterplot", "x_col": "x-axis column", "y_col": "y-axis column", "options": {{"color": "red", "style": "dotted"}}}}
        """
        logging.info("Generating analysis plan...")
        plan = await self.llm.generate_json_response(prompt)
        logging.info(f"Generated Plan: {plan}")
        return plan

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and prepares a dataframe for analysis."""
        # Clean column names (e.g., "Rank[a]" -> "Rank")
        df.columns = [re.sub(r'\[.*?\]', '', col).strip() for col in df.columns]
        
        # Aggressively convert columns to numeric where possible
        for col in df.columns:
            if df[col].dtype == 'object':
                # Remove currency symbols, commas, and other non-numeric chars
                cleaned_series = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                # Convert to numeric, coercing errors to NaN (Not a Number)
                df[col] = pd.to_numeric(cleaned_series, errors='coerce')
        return df

    def _get_actual_column_name(self, df_columns: List[str], requested_name: str) -> str:
        """Finds the actual column name that best matches the requested name."""
        requested_lower = requested_name.lower()
        for col in df_columns:
            if requested_lower in col.lower():
                return col
        # Fallback if no partial match is found
        raise ValueError(f"Could not find a column in the data that matches '{requested_name}'. Available columns: {df_columns}")


    async def _execute_plan(self, plan: Dict) -> Dict:
        execution_results = {}
        dataframes = {} # Store loaded dataframes in memory

        for i, step in enumerate(plan.get("plan", [])):
            action, params = step.get("action"), step.get("params", {})
            step_key = f"step_{i+1}_{action}"
            logging.info(f"Executing step {i+1}: {action} with params {params}")
            
            try:
                if action == "scrape_website":
                    proxy_url = f"https://aipipe.org/proxy/{params['url']}"
                    async with httpx.AsyncClient() as client:
                        response = await client.get(proxy_url, follow_redirects=True, timeout=60.0)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    table = soup.find('table', {'class': 'wikitable'})
                    if table:
                        df = pd.read_html(io.StringIO(str(table)))[0]
                        df = self._clean_dataframe(df)
                        dataframes['scraped_data.csv'] = df
                        execution_results[step_key] = f"Scraped data. Columns: {df.columns.tolist()}"
                
                elif action == "run_duckdb_sql":
                    # The query from the LLM might use generic names like "Gross" or "Title"
                    query = params["query"]
                    # Assume the query is on the first available dataframe
                    df_name = next(iter(dataframes))
                    df = dataframes[df_name]
                    
                    # Dynamically replace generic column names with actual ones
                    for generic_name in ['Gross', 'Year', 'Rank', 'Peak', 'Title']:
                        try:
                            actual_name = self._get_actual_column_name(df.columns, generic_name)
                            query = re.sub(r'\b' + generic_name + r'\b', f'"{actual_name}"', query, flags=re.IGNORECASE)
                        except ValueError:
                            pass # Ignore if a column isn't found, the query might not use it
                    
                    logging.info(f"Executing rewritten SQL: {query}")
                    with duckdb.connect(database=':memory:') as con:
                        con.register('data_table', df)
                        result_df = con.execute(query).df()
                    execution_results[step_key] = result_df.to_dict(orient='records')

                elif action == "generate_plot":
                    df = dataframes[params["file_name"]]
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    # Find actual column names
                    x_col_actual = self._get_actual_column_name(df.columns, params["x_col"])
                    y_col_actual = self._get_actual_column_name(df.columns, params["y_col"])
                    
                    ax.scatter(df[x_col_actual], df[y_col_actual], alpha=0.6)
                    
                    if "options" in params:
                        clean_df = df[[x_col_actual, y_col_actual]].dropna()
                        if not np.issubdtype(clean_df[x_col_actual].dtype, np.number) or not np.issubdtype(clean_df[y_col_actual].dtype, np.number):
                             raise TypeError("Plotting requires numeric data.")
                        
                        z = np.polyfit(clean_df[x_col_actual], clean_df[y_col_actual], 1)
                        p = np.poly1d(z)
                        ax.plot(clean_df[x_col_actual], p(clean_df[x_col_actual]), color=params["options"].get("color", "red"), linestyle=params["options"].get("style", "dotted"))
                    
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
        Use the 'Execution Results' to find the values. If a step has an error, reflect that in your answer.

        **Original Questions:**
        {questions}

        **Execution Results:**
        {json.dumps(results, indent=2)}

        Provide only the final, formatted JSON output. Do not include explanations.
        """
        logging.info("Synthesizing final answer...")
        return await self.llm.generate_json_response(prompt)

    async def analyze(self, questions: str, files: List[UploadFile]) -> Any:
        context = {"files": [f.filename for f in files if f.filename != "questions.txt"]}
        urls = re.findall(r'http[s]?://\S+', questions)
        if urls: context["urls_in_question"] = urls
        
        # Save uploaded files (if any)
        for file in files:
            if file.filename != "questions.txt":
                file_path = os.path.join(self.temp_dir, file.filename)
                async with aiofiles.open(file_path, 'wb') as f: await f.write(await file.read())
                self.file_paths[file.filename] = file_path
            
        plan = await self._create_plan(questions, context)
        results = await self._execute_plan(plan)
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

