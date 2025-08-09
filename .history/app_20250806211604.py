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
    logging.warning("AIPIPE_TOKEN environment variable not set. The application will run, but AI calls will fail.")

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
            raise ValueError("AI Pipe token is missing. Cannot make a request.")
            
        messages = [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": messages, "response_format": {"type": "json_object"}}
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        
        try:
            response = await self.client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            return json.loads(content)
        except (httpx.RequestError, json.JSONDecodeError, KeyError, TypeError) as e:
            logging.error(f"Error generating response from AI Pipe: {e}")
            logging.error(f"Response body from AI Pipe: {response.text if 'response' in locals() else 'No response'}")
            raise

# --- Core Data Analyst Agent ---
class DataAnalystAgent:
    def __init__(self, llm_client: AIPipeClient):
        self.llm = llm_client
        self.temp_dir = tempfile.mkdtemp()
        self.file_paths = {}

    # --- TWEAKABLE PART 1: The Planning Prompt ---
    # This prompt is now more explicit about using the scraped data file.
    async def _create_plan(self, questions: str, context: Dict) -> Dict:
        prompt = f"""
        You are an expert data analyst agent. Your primary job is to create a flawless, step-by-step JSON execution plan.
        Carefully analyze the user's questions and the available context to formulate the plan.

        **User's Questions:**
        {questions}
        
        **Available Context & Files:**
        {json.dumps(context, indent=2)}

        **Your Task:**
        Create a JSON object with a key "plan". This key holds a list of action steps.
        Each step MUST be a JSON object with "action" and "params" keys.
        If data is scraped from a website, it will be saved as "scraped_data.csv". All subsequent steps like "analyze_dataframe" or "generate_plot" MUST refer to this filename.

        **Supported Actions & their Params:**
        - "scrape_website": {{"url": "The full URL to scrape from the user's question"}}
        - "run_duckdb_sql": {{"query": "The exact SQL query to run. You can query tables created from scraped or uploaded files."}}
        - "analyze_dataframe": {{"file_name": "The name of the CSV file (e.g., 'scraped_data.csv')", "operations": ["list of operations like 'correlation', 'count_rows_condition'"]}}
        - "generate_plot": {{"file_name": "The name of the CSV file (e.g., 'scraped_data.csv')", "plot_type": "e.g., 'scatterplot'", "x_col": "The column name for the x-axis", "y_col": "The column name for the y-axis", "options": {{"color": "red", "style": "dotted"}}}}
        - "answer_from_context": {{"question": "A specific question that can be answered directly from the text"}}

        Now, create the plan for the user's request.
        """
        logging.info("Generating analysis plan via AI Pipe...")
        plan = await self.llm.generate_json_response(prompt)
        logging.info(f"Generated Plan: {plan}")
        return plan

    # --- TWEAKABLE PART 2: The Execution Logic ---
    # This logic now uses a single, shared DuckDB connection to maintain state.
    async def _execute_plan(self, plan: Dict) -> Dict:
        execution_results = {}
        # Establish a single, in-memory DuckDB connection for the entire plan
        with duckdb.connect(database=':memory:', read_only=False) as con:
            con.execute("INSTALL httpfs; LOAD httpfs;")
            
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
                            df.columns = [re.sub(r'\[.*?\]', '', col).strip() for col in df.columns]
                            
                            # Save to a temp file AND register it as a table in DuckDB
                            temp_path = os.path.join(self.temp_dir, "scraped_data.csv")
                            df.to_csv(temp_path, index=False)
                            self.file_paths["scraped_data.csv"] = temp_path
                            con.register('scraped_data', df) # Register dataframe as a table
                            
                            execution_results[step_key] = f"Scraped and saved as scraped_data.csv. Registered as table 'scraped_data'. Columns: {df.columns.tolist()}"
                    
                    elif action == "run_duckdb_sql":
                        # Now this action uses the shared connection where tables are registered
                        result_df = con.execute(params["query"]).df()
                        execution_results[step_key] = result_df.to_dict(orient='records')

                    elif action == "analyze_dataframe":
                        file_name = params["file_name"]
                        if file_name not in self.file_paths:
                            raise FileNotFoundError(f"File '{file_name}' not found. Was it uploaded or scraped?")
                        
                        df = pd.read_csv(self.file_paths[file_name])
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.,-]', '', regex=True).str.replace(',', ''), errors='coerce')
                        
                        analysis = {}
                        for op in params.get("operations", []):
                            if op == "correlation" and "x_col" in params and "y_col" in params:
                                analysis['correlation'] = df[params["x_col"]].corr(df[params["y_col"]])
                            if op == "count_rows_condition" and "column" in params and "value" in params:
                                 analysis['count'] = len(df[df[params["column"]] < params["value"]])
                        execution_results[step_key] = analysis

                    elif action == "generate_plot":
                        file_name = params["file_name"]
                        if file_name not in self.file_paths:
                            raise FileNotFoundError(f"File '{file_name}' not found for plotting.")
                        
                        df = pd.read_csv(self.file_paths[file_name])
                        fig, ax = plt.subplots(figsize=(8, 5))
                        x_col, y_col = params["x_col"], params["y_col"]
                        ax.scatter(df[x_col], df[y_col], alpha=0.6)
                        if "options" in params:
                            clean_df = df[[x_col, y_col]].dropna()
                            z = np.polyfit(clean_df[x_col], clean_df[y_col], 1)
                            p = np.poly1d(z)
                            ax.plot(clean_df[x_col], p(clean_df[x_col]), color=params["options"].get("color", "red"), linestyle=params["options"].get("style", "dotted"))
                        ax.set_xlabel(x_col); ax.set_ylabel(y_col); plt.grid(True, alpha=0.3)
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

    # --- TWEAKABLE PART 3: The Synthesis Prompt ---
    # This prompt remains the same, but it will now receive better data.
    async def _synthesize_answer(self, questions: str, results: Dict) -> Any:
        prompt = f"""
        You are a synthesizer agent. Your job is to construct the final answer based on the user's original request and the results from the execution plan.
        You MUST format your response exactly as requested in the original questions (e.g., a JSON array or a JSON object).
        Use the provided 'Execution Results' to find the values for your answer. Do not make up information. If a step resulted in an error, reflect that in your answer.

        **Original Questions:**
        {questions}

        **Execution Results (use these values to construct the answer):**
        {json.dumps(results, indent=2)}

        Provide only the final, formatted JSON output. Do not include any other text, markdown, or explanations.
        """
        logging.info("Synthesizing final answer...")
        return await self.llm.generate_json_response(prompt)

    async def analyze(self, questions: str, files: List[UploadFile]) -> Any:
        context = {"files": []}
        for file in files:
            if file.filename != "questions.txt":
                file_path = os.path.join(self.temp_dir, file.filename)
                async with aiofiles.open(file_path, 'wb') as f: await f.write(await file.read())
                self.file_paths[file.filename] = file_path
                context["files"].append(file.filename)
        
        urls = re.findall(r'http[s]?://\S+', questions)
        if urls: context["urls_in_question"] = urls
            
        sql_match = re.search(r"```sql(.*?)```", questions, re.DOTALL)
        if sql_match: context["sql_query_in_question"] = sql_match.group(1).strip()
            
        plan = await self._create_plan(questions, context)
        results = await self._execute_plan(plan)
        return await self._synthesize_answer(questions, results)

# --- FastAPI App ---
app = FastAPI(title="AI Pipe Data Analyst Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
@app.get("/health")
