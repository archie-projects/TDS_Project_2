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
import uvicorn
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
AIPIPE_TOKEN = os.getenv('AIPIPE_TOKEN')
AIPipe_BASE_URL = "https://aipipe.org/openrouter/v1" # Using the OpenRouter proxy is versatile
AIPipe_MODEL = os.getenv('AIPipe_MODEL', 'openai/gpt-4o-mini') # A powerful and free model

if not AIPIPE_TOKEN:
    raise ValueError("AIPIPE_TOKEN environment variable not set.")

# --- LLM Client for AI Pipe ---
class AIPipeClient:
    def __init__(self, token: str, base_url: str, model: str):
        self.token = token
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=180.0) # 3 minute timeout
        logging.info(f"Initialized AI Pipe client for model: {model}")

    async def generate_json_response(self, prompt: str) -> Dict[str, Any]:
        """Generates a JSON response from a prompt via AI Pipe."""
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

# --- Core Data Analyst Agent (Unchanged Logic) ---
class DataAnalystAgent:
    def __init__(self, llm_client: AIPipeClient):
        self.llm = llm_client
        self.temp_dir = tempfile.mkdtemp()
        self.file_paths = {}

    async def _create_plan(self, questions: str, context: Dict) -> Dict:
        # This prompt asks the LLM to act as the 'brain'
        prompt = f"""
        You are an expert data analyst agent. Create a step-by-step JSON execution plan to answer the user's questions based on the provided context.
        Your output MUST be a JSON object with a key "plan", which is a list of steps.

        **User's Questions:**
        {questions}
        
        **Available Context & Files:**
        {json.dumps(context, indent=2)}

        **Supported Actions & Params:**
        - "scrape_website": {{"url": "URL to scrape"}}
        - "run_duckdb_sql": {{"query": "SQL query"}}
        - "analyze_dataframe": {{"file_name": "CSV file name", "operations": ["list of operations like 'correlation'"]}}
        - "generate_plot": {{"file_name": "CSV file name", "plot_type": "scatterplot", "x_col": "column", "y_col": "column", "options": {{"color": "red", "style": "dotted"}}}}
        - "answer_from_context": {{"question": "The specific question"}}
        """
        logging.info("Generating analysis plan via AI Pipe...")
        plan = await self.llm.generate_json_response(prompt)
        logging.info(f"Generated Plan: {plan}")
        return plan

    async def _execute_plan(self, plan: Dict) -> Dict:
        execution_results = {}
        for i, step in enumerate(plan.get("plan", [])):
            action, params = step.get("action"), step.get("params", {})
            step_key = f"step_{i+1}_{action}"
            logging.info(f"Executing step {i+1}: {action}")
            
            try:
                if action == "scrape_website":
                    # Use the AI Pipe proxy for reliable scraping, bypassing CORS issues
                    proxy_url = f"https://aipipe.org/proxy/{params['url']}"
                    async with httpx.AsyncClient() as client:
                        response = await client.get(proxy_url, follow_redirects=True)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    tables = soup.find_all('table', {'class': 'wikitable'})
                    if tables:
                        df = pd.read_html(io.StringIO(str(tables[0])))[0]
                        df.columns = [re.sub(r'\[.*?\]', '', col).strip() for col in df.columns]
                        temp_path = os.path.join(self.temp_dir, "scraped_data.csv")
                        df.to_csv(temp_path, index=False)
                        self.file_paths["scraped_data.csv"] = temp_path
                        execution_results[step_key] = f"Scraped and saved as scraped_data.csv. Shape: {df.shape}"
                
                elif action == "run_duckdb_sql":
                    with duckdb.connect() as con:
                        df = con.execute(params["query"]).df()
                    execution_results[step_key] = df.to_dict(orient='records')

                elif action == "analyze_dataframe":
                    df = pd.read_csv(self.file_paths[params["file_name"]])
                    df.columns = [re.sub(r'\[.*?\]', '', col).strip() for col in df.columns]
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
                    analysis = {}
                    if "correlation" in params["operations"]:
                        numeric_cols = df.select_dtypes(include=np.number).columns
                        if len(numeric_cols) >= 2:
                            analysis['correlation'] = df[numeric_cols[0]].corr(df[numeric_cols[1]])
                    execution_results[step_key] = analysis

                elif action == "generate_plot":
                    df = pd.read_csv(self.file_paths[params["file_name"]])
                    fig, ax = plt.subplots(figsize=(8, 5))
                    x_col, y_col = params["x_col"], params["y_col"]
                    ax.scatter(df[x_col], df[y_col], alpha=0.6)
                    if "options" in params:
                        z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
                        p = np.poly1d(z)
                        ax.plot(df[x_col], p(df[x_col]), color=params["options"].get("color", "red"), linestyle=params["options"].get("style", "dotted"))
                    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                    plt.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=90)
                    plt.close(fig)
                    base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                    execution_results[step_key] = f"data:image/png;base64,{base64_img}"
            
            except Exception as e:
                logging.error(f"Error executing step {i+1} ({action}): {e}")
                execution_results[step_key] = {"error": str(e)}
        return execution_results

    async def _synthesize_answer(self, questions: str, results: Dict) -> Any:
        prompt = f"""
        Based on the user's questions and the executed plan results, generate the final answer.
        You MUST format your response exactly as requested in the original questions. Do not add any explanation.

        **Original Questions:**
        {questions}

        **Execution Results (use these values to construct the answer):**
        {json.dumps(results, indent=2)}

        Provide only the final, formatted JSON output.
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
        logging.error(f"Unhandled error in analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": f"AI Pipe Data Analyst Agent is running with model {AIPipe_MODEL}."}6666666
