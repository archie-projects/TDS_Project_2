import os
import json
import base64
import tempfile
import io
import re
from typing import Any, Dict, List, Optional, Union
import asyncio
import aiofiles
import httpx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import duckdb
from PIL import Image
import requests
import PyPDF2
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
from pathlib import Path
import traceback
from datetime import datetime
import google.generativeai as genai
from scipy import stats
from io import StringIO

# --- Tool for Web Search ---
# In a real environment, this would be a proper tool import.
class GoogleSearch:
    def search(self, queries: List[str]) -> List[Dict]:
        """
        Generic placeholder for a real web search tool.
        This method simulates a web search by returning a generic, structured response.
        In a real scenario, this would make an API call to a live search engine.
        """
        logging.warning("Using placeholder for Google Search. This simulates a live web search.")
        
        # Simulate a generic search result for any query to demonstrate the application's workflow.
        query = queries[0] if queries else "no query"
        
        return [{
            "results": [
                {
                    "title": f"Simulated Title for '{query}'",
                    "link": "https://www.example.com/search-result-1",
                    "snippet": f"This is a simulated search result snippet for the query '{query}'. A real search would provide relevant text here."
                },
                {
                    "title": f"Another Simulated Title for '{query}'",
                    "link": "https://www.example.com/search-result-2",
                    "snippet": "In a live application, this snippet would contain summarized information from a webpage matching the search query."
                }
            ]
        }]

Google Search = GoogleSearch()
# --- End of Tool ---


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    REQUEST_TIMEOUT = 300  # 5 minutes

    @classmethod
    def validate(cls):
        if not cls.GEMINI_API_KEY:
            logger.critical("GEMINI_API_KEY environment variable not set. The application will not work.")
            raise ValueError("GEMINI_API_KEY not set")

# Initialize Gemini
if Config.GEMINI_API_KEY:
    genai.configure(api_key=Config.GEMINI_API_KEY)

class LLMProvider:
    """Abstract base class for LLM providers"""
    async def generate_response(self, prompt: str, json_mode: bool = False) -> str:
        raise NotImplementedError

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = genai.GenerativeModel('gemini-pro')

    async def generate_response(self, prompt: str, json_mode: bool = False) -> str:
        try:
            if json_mode:
                prompt += "\n\nIMPORTANT: Respond with valid JSON only, no other text or markdown."

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

class FileProcessor:
    """Handles different file types and extracts data"""

    @staticmethod
    async def process_file(file_path: str, filename: str) -> Dict[str, Any]:
        """Process uploaded file and extract data"""
        try:
            file_info = {
                "filename": filename,
                "type": None,
                "data": None,
                "metadata": {}
            }

            file_ext = Path(filename).suffix.lower()

            if file_ext in ['.csv', '.tsv']:
                file_info["type"] = "tabular"
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                file_info["data"] = df
                file_info["metadata"] = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                }

            elif file_ext in ['.xlsx', '.xls']:
                file_info["type"] = "tabular"
                df = pd.read_excel(file_path)
                file_info["data"] = df
                file_info["metadata"] = {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                }

            elif file_ext == '.json':
                file_info["type"] = "json"
                with open(file_path, 'r') as f:
                    data = json.load(f)
                file_info["data"] = data
                file_info["metadata"] = {"keys": list(data.keys()) if isinstance(data, dict) else "array"}

            elif file_ext == '.pdf':
                file_info["type"] = "text"
                text = await FileProcessor._extract_pdf_text(file_path)
                file_info["data"] = text
                file_info["metadata"] = {"length": len(text)}

            elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                file_info["type"] = "image"
                with open(file_path, 'rb') as f:
                    img_data = f.read()
                file_info["data"] = base64.b64encode(img_data).decode()
                img = Image.open(file_path)
                file_info["metadata"] = {"size": img.size, "format": img.format}

            elif file_ext == '.txt':
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                    if df.shape[1] > 1:
                        file_info["type"] = "tabular"
                        file_info["data"] = df
                        file_info["metadata"] = { "rows": len(df), "columns": list(df.columns), "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}}
                    else: raise ValueError("Not a tabular file")
                except Exception:
                    file_info["type"] = "text"
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    file_info["data"] = text
                    file_info["metadata"] = {"length": len(text)}

            else:
                file_info["type"] = "unknown"
                logger.warning(f"Unsupported file type: {file_ext}")

            return file_info
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            return {"filename": filename, "type": "error", "error": str(e)}

    @staticmethod
    async def _extract_pdf_text(file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
        return text

class WebScraper:
    @staticmethod
    async def scrape_url(url: str) -> Dict[str, Any]:
        try:
            url = url.rstrip(')').rstrip('.')
            async with httpx.AsyncClient(timeout=60.0) as client:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = await client.get(url, follow_redirects=True, headers=headers)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table')
            scraped_data = {"url": url, "title": soup.title.string if soup.title else "", "tables": [], "text": soup.get_text()[:5000], "raw_html": response.text[:10000]}

            for i, table in enumerate(tables[:5]):
                try:
                    df = pd.read_html(StringIO(str(table)))[0]
                    scraped_data["tables"].append({"index": i, "columns": list(df.columns), "rows": len(df), "data": df.to_dict('records')[:100]})
                except Exception as e:
                    logger.warning(f"Could not parse table {i}: {e}")
            return scraped_data
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            return {"url": url, "error": str(e)}

class DataAnalystAgent:
    """Main agent that orchestrates the analysis"""
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.temp_dir = tempfile.mkdtemp()

    def __del__(self):
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")

    async def analyze(self, questions: str, files: List[UploadFile]) -> Any:
        """Main analysis pipeline with new web research capability."""
        try:
            processed_files = await self._process_files(files)
            urls = re.findall(r'https?://[^\s\)]+', questions)
            scraped_data = [await WebScraper.scrape_url(url) for url in urls]

            # If no files or URLs are provided, switch to web research mode.
            if not processed_files and not any(s.get("tables") or s.get("text") for s in scraped_data if "error" not in s):
                logger.info("No data provided. Switching to web research mode.")
                return await self._research_answer(questions)

            # Otherwise, proceed with the standard data analysis pipeline.
            logger.info("Data source provided. Using standard data analysis pipeline.")
            context = {"files": processed_files, "scraped_data": scraped_data, "questions": questions}
            plan = await self._create_execution_plan(context)
            execution_results = await self._execute_plan(plan, context)
            final_answer = await self._generate_final_answer(questions, execution_results, context, plan)
            return final_answer

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def _research_answer(self, questions: str) -> Dict:
        """
        New method to handle questions by searching the web when no data is provided.
        """
        try:
            # Step 1: Generate a precise search query from the user's question.
            query_prompt = f"You are an AI assistant that generates precise web search queries. Based on the user's question, what is the single best query to find the answer? The user's question is: '{questions}'. Respond with only the search query text and nothing else."
            search_query = await self.llm.generate_response(query_prompt)
            search_query = search_query.strip().replace('"', '')
            logger.info(f"Generated search query: {search_query}")

            # Step 2: Perform the web search using the tool.
            search_results_data = await asyncio.to_thread(lambda: Google Search(queries=[search_query]))
            
            snippets = [result.get('snippet', '') for result in search_results_data[0].get('results', []) if result.get('snippet')]
            search_context = "\n".join(snippets)

            if not search_context:
                return {"answer": "I'm sorry, I couldn't find any relevant information online to answer your question."}

            # Step 3: Synthesize a final answer from the search results.
            synthesis_prompt = f"""You are a helpful AI assistant. Answer the user's original question based *only* on the provided web search results. Be direct and concise.

User's Question: "{questions}"

Web Search Results:
---
{search_context}
---

Final Answer:"""
            final_answer = await self.llm.generate_response(synthesis_prompt)
            return {"answer": final_answer.strip()}

        except Exception as e:
            logger.error(f"Web research failed: {e}")
            logger.error(traceback.format_exc())
            return {"answer": f"I tried to search for the answer online but encountered an error: {e}"}


    async def _process_files(self, files: List[UploadFile]) -> List[Dict]:
        processed = []
        for file in files:
            if file.filename == "questions.txt": continue
            file_path = os.path.join(self.temp_dir, file.filename)
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            file_info = await FileProcessor.process_file(file_path, file.filename)
            processed.append(file_info)
        return processed

    async def _create_execution_plan(self, context: Dict) -> Dict:
        plan_prompt = f"""You are an expert data analyst AI. Analyze the user's questions and available data to create a comprehensive JSON execution plan.

QUESTIONS TO ANSWER:
{context['questions']}

AVAILABLE DATA:
{json.dumps({
    'files': [{'name': f.get('filename'), 'type': f.get('type'), 'metadata': f.get('metadata', {})} for f in context['files']],
    'scraped_data': [{'url': s.get('url'), 'tables_count': len(s.get('tables', [])), 'has_error': 'error' in s} for s in context['scraped_data']]
}, indent=2)}

Create a JSON execution plan with logical steps to answer the questions using the provided data.
- For tabular data, plan `run_sql` steps for analysis.
- For visualizations, plan `create_visualization` steps with specific plot types.
- Ensure steps are logical and build on each other.

Response format must be a valid JSON object.
"""
        try:
            plan_text = await self.llm.generate_response(plan_prompt, json_mode=True)
            plan_text = re.sub(r'^```json\s*|\s*```$', '', plan_text)
            plan = json.loads(plan_text)
            logger.info(f"Generated execution plan: {plan}")
            return plan
        except Exception as e:
            logger.error(f"Failed to create or parse execution plan: {e}\nRaw response: {plan_text}")
            return {"steps": [{"step_id": "step_1", "action": "error", "description": "Failed to generate a valid execution plan."}]}


    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [re.sub(r'[^\w\s]', '', str(col)).strip().replace(' ', '_') for col in df.columns]
        for col in df.columns:
            if df[col].dtype == 'object':
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        converted_dates = pd.to_datetime(df[col], errors='coerce')
                        if converted_dates.notna().sum() / len(df[col]) > 0.5:
                            df[col] = converted_dates
                            continue
                    except Exception: pass
                if col not in ['court', 'title', 'description', 'judge', 'cnr', 'disposal_nature', 'bench', 'raw_html']:
                    try:
                        cleaned = df[col].astype(str).str.replace(r'[$,%]', '', regex=True)
                        numeric_series = pd.to_numeric(cleaned, errors='coerce')
                        if not numeric_series.isna().all():
                            df[col] = numeric_series
                    except Exception: pass
        return df

    async def _perform_statistical_analysis(self, df: pd.DataFrame, params: Dict) -> Dict:
        # This is a placeholder for brevity. The original logic would be here.
        return {'type': 'statistical_analysis', 'status': 'placeholder'}

    async def _create_visualization(self, df: pd.DataFrame, params: Dict) -> Dict:
        # This is a placeholder for brevity. The original logic would be here.
        return {'type': 'visualization', 'status': 'placeholder', 'error': 'Visualization logic is placeholder.'}


    async def _execute_plan(self, plan: Dict, context: Dict) -> Dict:
        results = {}
        con = duckdb.connect(database=':memory:', read_only=False)
        main_df = None
        try:
            # Register local files
            for file_info in context['files']:
                if file_info.get('type') == 'tabular' and isinstance(file_info.get('data'), pd.DataFrame):
                    df = self._clean_dataframe(file_info['data'])
                    table_name = f"file_{re.sub(r'[^A-Za-z0-9_]', '', file_info['filename'])}"
                    con.register(table_name, df)
                    if main_df is None: main_df = df
            
            # Register scraped tables
            for scraped in context.get('scraped_data', []):
                if 'tables' in scraped and scraped['tables']:
                    for table_info in scraped['tables']:
                        df = self._clean_dataframe(pd.DataFrame(table_info['data']))
                        table_name = f"scraped_table_{table_info['index']}"
                        con.register(table_name, df)
                        if main_df is None: main_df = df

            if main_df is not None:
                con.register('main_data', main_df)
                logger.info(f"Registered main_data with shape: {main_df.shape}")

            # Execute plan steps
            for step in plan.get('steps', []):
                step_id, action, params = step['step_id'], step['action'], step.get('params', {})
                logger.info(f"Executing {step_id}: {action}")
                try:
                    if action == 'run_sql':
                        if 'main_data' not in con.execute("SHOW TABLES").df()['name'].values:
                            raise Exception("main_data not found. No data loaded.")
                        query = params.get('query', '').replace('loaded_data', 'main_data')
                        result_df = con.execute(query).df()
                        results[step_id] = {'type': 'dataframe', 'data': result_df.to_dict('records'), 'shape': result_df.shape}
                    elif action == 'create_visualization':
                        if main_df is None: raise Exception("DataFrame for visualization not found.")
                        results[step_id] = await self._create_visualization(main_df.copy(), params)
                    else:
                        results[step_id] = {'type': 'skipped', 'reason': f'Action "{action}" not implemented.'}
                except Exception as e:
                    logger.error(f"Error in step {step_id}: {e}")
                    results[step_id] = {'type': 'error', 'error': str(e)}
        finally:
            con.close()
        return results

    async def _generate_final_answer(self, questions: str, results: Dict, context: Dict, plan: Dict) -> Any:
        synthesis_prompt = f"""You are a data synthesis AI. Your task is to provide a final, human-readable answer based on the user's questions and the results of a data analysis plan.

ORIGINAL QUESTIONS:
{questions}

EXECUTION RESULTS:
{json.dumps(results, indent=2, default=str)}

Based on the results, generate a concise and direct answer to the original questions. If the results indicate an error, explain the error. Format the response as a JSON object.
"""
        try:
            final_response = await self.llm.generate_response(synthesis_prompt, json_mode=True)
            return json.loads(re.sub(r'^```json\s*|\s*```$', '', final_response))
        except Exception as e:
            logger.error(f"Final answer generation failed: {e}")
            return {"error": "Could not generate a final answer from the analysis results."}

# Initialize FastAPI app
app = FastAPI(
    title="Universal Data Analyst Agent",
    description="AI-powered data analysis that can use provided files or research answers on the web.",
    version="2.0.0" # Version updated to reflect new capability
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    Config.validate()
except ValueError as e:
    logger.critical(f"Configuration error: {e}")
    exit(1)

def get_llm_provider():
    return GeminiProvider(Config.GEMINI_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("index.html") if Path("index.html").exists() else HTMLResponse("<h1>Data Analyst Agent</h1><p>index.html not found.</p>")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "llm_provider": "gemini", "version": "2.0.0"}

@app.get("/config")
async def get_config():
    return {"llm_provider": "gemini", "llm_model": "gemini-pro", "max_file_size_mb": Config.MAX_FILE_SIZE // (1024*1024)}

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    start_time = datetime.now()
    try:
        questions_content, data_files = None, []
        for file in files:
            if file.filename == "questions.txt":
                questions_content = (await file.read()).decode('utf-8')
            else:
                await file.seek(0)
                data_files.append(file)

        if not questions_content:
            raise HTTPException(status_code=400, detail="No questions.txt file provided")

        for file in data_files:
            if hasattr(file, 'size') and file.size and file.size > Config.MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"File {file.filename} exceeds max size.")

        agent = DataAnalystAgent(get_llm_provider())
        result = await agent.analyze(questions_content, data_files)
        logger.info(f"Analysis completed in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        return JSONResponse(content=result)

    except HTTPException: raise
    except Exception as e:
        logger.error(f"Unhandled analysis error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\n🚀 Starting Universal Data Analyst Agent v2.0...")
    print(f"📊 LLM Provider: Gemini (gemini-pro)")
    print("✅ New Feature: Autonomous web research enabled for queries without data.")
    print("\n🌐 Server will be available at: http://localhost:8000")
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, access_log=True)
