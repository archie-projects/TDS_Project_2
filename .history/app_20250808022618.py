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
import shutil
from urllib.parse import quote, unquote

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    AIPIPE_TOKEN = os.getenv('AIPIPE_TOKEN')
    AIPIPE_MODEL = os.getenv('AIPIPE_MODEL', 'openai/gpt-4o-mini')
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')
    DEFAULT_LLM_PROVIDER = os.getenv('DEFAULT_LLM_PROVIDER', 'aipipe')
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    REQUEST_TIMEOUT = 300  # 5 minutes
    
    @classmethod
    def validate(cls):
        if cls.DEFAULT_LLM_PROVIDER == 'gemini' and not cls.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set")
        if cls.DEFAULT_LLM_PROVIDER == 'aipipe' and not cls.AIPIPE_TOKEN:
            logger.warning("AIPIPE_TOKEN not set")

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
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    async def generate_response(self, prompt: str, json_mode: bool = False) -> str:
        try:
            if json_mode:
                prompt += "\n\nIMPORTANT: Respond with valid JSON only, no other text."
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

class AIPipeProvider(LLMProvider):
    def __init__(self, token: str, base_url: str = "https://aipipe.org/openrouter/v1", model: str = "openai/gpt-4o-mini"):
        self.token = token
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT)
        
    async def generate_response(self, prompt: str, json_mode: bool = False) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            payload = {"model": self.model, "messages": messages}
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            headers = {"Authorization": f"Bearer {self.token}"}
            response = await self.client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"AIPipe API error: {e}")
            raise

class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT)
        
    async def generate_response(self, prompt: str, json_mode: bool = False) -> str:
        try:
            if json_mode:
                prompt += "\n\nIMPORTANT: Respond with valid JSON only, no other text."
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            if json_mode:
                payload["format"] = "json"
            response = await self.client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

class FileProcessor:
    """Handles different file types and extracts data"""
    @staticmethod
    def _extract_pdf_text(file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
        return text

    @staticmethod
    async def process_file(file_path: str, filename: str) -> Dict[str, Any]:
        try:
            file_info = {"filename": filename, "type": None, "data": None, "metadata": {}}
            file_ext = Path(filename).suffix.lower()

            if file_ext in ['.csv', '.tsv']:
                file_info["type"] = "tabular"
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                file_info["data"] = df
            elif file_ext in ['.xlsx', '.xls']:
                file_info["type"] = "tabular"
                df = pd.read_excel(file_path)
                file_info["data"] = df
            elif file_ext == '.json':
                file_info["type"] = "json"
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    file_info["data"] = json.loads(await f.read())
            elif file_ext == '.pdf':
                file_info["type"] = "text"
                file_info["data"] = await asyncio.to_thread(FileProcessor._extract_pdf_text, file_path)
            elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                file_info["type"] = "image"
                async with aiofiles.open(file_path, 'rb') as f:
                    img_data = await f.read()
                file_info["data"] = base64.b64encode(img_data).decode()
                img = Image.open(file_path)
                file_info["metadata"] = {"size": img.size, "format": img.format}
            elif file_ext == '.txt':
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', sep=r'\s{2,}', engine='python')
                    if df.shape[1] > 1:
                        file_info["type"] = "tabular"
                        file_info["data"] = df
                    else:
                        raise ValueError("Single column, treat as text")
                except Exception:
                    file_info["type"] = "text"
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        file_info["data"] = await f.read()
            else:
                file_info["type"] = "unknown"
                logger.warning(f"Unsupported file type: {file_ext}")
                return file_info

            if file_info["type"] == "tabular":
                df = file_info["data"]
                file_info["metadata"] = {
                    "rows": len(df), "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                }
            elif file_info["type"] == "text":
                file_info["metadata"] = {"length": len(file_info["data"])}
            elif file_info["type"] == "json":
                data = file_info["data"]
                file_info["metadata"] = {"keys": list(data.keys()) if isinstance(data, dict) else "array"}

            return file_info
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}\n{traceback.format_exc()}")
            return {"filename": filename, "type": "error", "error": str(e)}

class WebScraper:
    """Handles web scraping with various strategies"""
    @staticmethod
    def _clean_url(url: str) -> str:
        """Clean and properly encode URL"""
        url = url.strip().rstrip('.,)]}').strip()
        url = url.replace('`', '')
        if '%' in url and not url.endswith('%60'):
            return url
        if '://' in url:
            scheme, rest = url.split('://', 1)
            if '/' in rest:
                domain, path = rest.split('/', 1)
                path = quote(path, safe='/:?#[]@!$&\'()*+,;=')
                url = f"{scheme}://{domain}/{path}"
        return url

    @staticmethod
    async def scrape_url(url: str, proxy: bool = True) -> Dict[str, Any]:
        try:
            clean_url = WebScraper._clean_url(url)
            scrape_url = f"https://aipipe.org/proxy/{clean_url}" if proxy else clean_url
            
            logger.info(f"Scraping URL: {clean_url} via proxy: {proxy}")
            
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(scrape_url)
                response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            scraped_data = {
                "url": clean_url, 
                "title": soup.title.string if soup.title else "",
                "tables": [], 
                "text": soup.get_text(separator='\n', strip=True)[:5000]
            }
            
            for i, table in enumerate(soup.find_all('table')[:5]):
                try:
                    df = pd.read_html(StringIO(str(table)))[0]
                    df.columns = [str(col).strip() for col in df.columns]
                    scraped_data["tables"].append({
                        "index": i, 
                        "columns": list(df.columns), 
                        "rows": len(df),
                        "data": df.head(100).to_dict('records')
                    })
                except Exception as e:
                    logger.warning(f"Could not parse table {i} from {clean_url}: {e}")
            
            return scraped_data
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            if proxy:
                logger.info(f"Retrying {url} without proxy...")
                return await WebScraper.scrape_url(url, proxy=False)
            return {"url": url, "error": str(e)}

class DataAnalystAgent:
    """Main agent that orchestrates the analysis"""
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.temp_dir = tempfile.mkdtemp()
    
    async def analyze(self, questions: str, files: List[UploadFile]) -> Any:
        try:
            processed_files_task = self._process_files(files)
            urls = re.findall(r'https?://[^\s()<>]+', questions)
            scraped_data_task = asyncio.gather(*(WebScraper.scrape_url(url) for url in urls)) if urls else asyncio.sleep(0, result=[])
            
            processed_files, scraped_data = await asyncio.gather(processed_files_task, scraped_data_task)
            
            context = {"files": processed_files, "scraped_data": scraped_data, "questions": questions}
            plan = await self._create_execution_plan(context)
            execution_results = await self._execute_plan(plan, context)
            final_answer = await self._generate_final_answer(questions, execution_results, context)
            return final_answer
        except Exception as e:
            logger.error(f"Analysis failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def _process_files(self, files: List[UploadFile]) -> List[Dict]:
        async def process_single_file(file: UploadFile):
            if file.filename == "questions.txt": return None
            file_path = os.path.join(self.temp_dir, file.filename)
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(await file.read())
            return await FileProcessor.process_file(file_path, file.filename)
        results = await asyncio.gather(*(process_single_file(f) for f in files))
        return [res for res in results if res is not None]

    async def _create_execution_plan(self, context: Dict) -> Dict:
        simple_context = {
            'files': [{'name': f.get('filename'), 'type': f.get('type'), 'metadata': f.get('metadata')} for f in context['files']],
            'scraped_data': []
        }
        for s in context['scraped_data']:
            if 'error' not in s:
                simple_context['scraped_data'].append({
                    'url': s.get('url'), 
                    'title': s.get('title', ''),
                    'tables_count': len(s.get('tables', [])),
                    'has_text': bool(s.get('text'))
                })
        
        plan_prompt = f"""You are an expert data analyst AI. Create a JSON execution plan to answer the user's questions based on the available data.
        
User Questions:
{context['questions']}

Available Data Context:
{json.dumps(simple_context, indent=2)}

Execution Plan Format:
- The plan must be a single valid JSON object with a "steps" array.
- Each step must have "step_id", "action", "description", and "params".
- Available actions: "extract_info", "run_sql", "statistical_analysis", "create_visualization"
- For web data, use "extract_info" to parse specific information from scraped content.
- For SQL queries, use proper SQL syntax (e.g., WHERE amount > 500, not WHERE amount > 500_crore).

Respond with the JSON plan only.
"""
        try:
            plan_text = await self.llm.generate_response(plan_prompt, json_mode=True)
            plan_text = re.sub(r'^```json\s*|\s*```$', '', plan_text, flags=re.I)
            plan = json.loads(plan_text)
            logger.info(f"Generated execution plan: {json.dumps(plan, indent=2)}")
            return plan
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse plan JSON: {e}\nRaw response: {plan_text}")
            return {
                "steps": [{"step_id": 1, "action": "extract_info", "description": "Extract info from available data", "params": {"extract_all": True}}]
            }

    async def _execute_plan(self, plan: Dict, context: Dict) -> Dict:
        results, main_df = {}, None
        con = duckdb.connect(database=':memory:')
        try:
            con.execute("INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;")
            for f in context['files']:
                if f.get('type') == 'tabular' and isinstance(f.get('data'), pd.DataFrame):
                    df, name = self._clean_dataframe(f['data']), f"tbl_{Path(f['filename']).stem}"
                    con.register(name, df)
                    if main_df is None: main_df = df; con.register('main_data', df)
            for i, s in enumerate(context['scraped_data']):
                if 'error' not in s:
                    for j, table_data in enumerate(s.get('tables', [])):
                        try:
                            df, name = self._clean_dataframe(pd.DataFrame(table_data['data'])), f"scraped_{i}_{j}"
                            con.register(name, df)
                            if main_df is None: main_df = df; con.register('main_data', df)
                            logger.info(f"Registered table {name} with {len(df)} rows")
                        except Exception as e:
                            logger.error(f"Error registering scraped table: {e}")

            for step in plan.get('steps', []):
                step_id, action, params = step['step_id'], step['action'], step.get('params', {})
                logger.info(f"Executing step {step_id}: {action}")
                try:
                    if action == 'extract_info':
                        extracted_info = {}
                        for i, s in enumerate(context['scraped_data']):
                            if 'error' not in s:
                                extracted_info[f"source_{i}"] = {"url": s.get('url'), "title": s.get('title'), "tables_count": len(s.get('tables', [])), "text_preview": s.get('text', '')[:500]}
                        results[step_id] = {'type': 'extraction', 'data': extracted_info}
                    elif action == 'run_sql' and params.get('query'):
                        query = params['query']
                        query = re.sub(r'\b(\d+)_crore\b', r'\1', query)
                        query = re.sub(r'\b(\d+)_lakh\b', r'\1', query)
                        try:
                            res_df = con.execute(query).df()
                            results[step_id] = {'type': 'dataframe', 'data': res_df.to_dict('records')}
                        except Exception as sql_error:
                            results[step_id] = {'type': 'error', 'error': f'SQL error: {str(sql_error)}'}
                    elif action in ['statistical_analysis', 'create_visualization']:
                        if main_df is None:
                             results[step_id] = {'type': 'error', 'error': 'No data available'}
                             continue
                        if action == 'statistical_analysis':
                            results[step_id] = await self._perform_statistical_analysis(main_df.copy(), params)
                        else:
                            results[step_id] = await self._create_visualization(main_df.copy(), params)
                except Exception as e:
                    logger.error(f"Error in step {step_id}: {e}\n{traceback.format_exc()}")
                    results[step_id] = {'type': 'error', 'error': str(e)}
        finally:
            con.close()
        return results

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)).strip('_') for col in df.columns]
        for col in df.columns:
            if df[col].dtype == 'object':
                if 'date' in col.lower():
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif df[col].astype(str).str.contains(r'[\d,₹]+', na=False).any():
                    try:
                        cleaned = df[col].astype(str).str.replace(r'[₹,]', '', regex=True).str.strip()
                        numeric_series = pd.to_numeric(cleaned, errors='coerce')
                        if not numeric_series.isna().all():
                            df[col] = numeric_series
                    except Exception: pass
        return df

    async def _perform_statistical_analysis(self, df: pd.DataFrame, params: Dict) -> Dict:
        # This is a placeholder for more complex analysis logic
        try:
            return {'type': 'summary', 'data': df.describe().to_dict()}
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

    async def _create_visualization(self, df: pd.DataFrame, params: Dict) -> Dict:
        # This is a placeholder for more complex visualization logic
        try:
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) < 2:
                return {'type': 'error', 'message': 'Not enough numeric columns for a plot.'}
            
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
            ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.7)
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            ax.set_title(f"Scatter Plot of {numeric_cols[1]} vs {numeric_cols[0]}")
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            return {'type': 'visualization', 'data': f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"}
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

    async def _generate_final_answer(self, questions: str, results: Dict, context: Dict) -> Any:
        result_summary = {}
        for step_id, result in results.items():
            result_summary[f"step_{step_id}"] = result

        synthesis_prompt = f"""Based on the user's questions and analysis results, provide a comprehensive answer.

User Questions:
{questions}

Analysis Results:
{json.dumps(result_summary, indent=2, default=str)}

Scraped Data Summary:
{json.dumps([{"url": s.get('url'), "title": s.get('title'), "has_data": 'error' not in s} for s in context['scraped_data']], indent=2)}

Instructions:
- Create a single JSON object as the final response.
- Answer each question based on the available analysis results.
- If data was successfully scraped, extract relevant information from it.
- For visualization results, include the full base64 data URI.
- If analysis failed, explain what went wrong.

Respond with the JSON object only.
"""
        try:
            response_text = await self.llm.generate_response(synthesis_prompt, json_mode=True)
            # FIX: Corrected regex and removed typo
            response_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.I)
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Final answer synthesis failed: {e}\nRaw response: {response_text if 'response_text' in locals() else 'No response'}")
            return {"error": "Failed to generate final answer", "raw_results": result_summary}

# --- FastAPI App Initialization ---

app = FastAPI(
    title="Universal Data Analyst Agent",
    description="An AI agent for data analysis, supporting files, web scraping, and visualizations.",
    version="1.2.1"
)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

Config.validate()

def get_llm_provider():
    provider = Config.DEFAULT_LLM_PROVIDER
    if provider == 'gemini' and Config.GEMINI_API_KEY: return GeminiProvider(Config.GEMINI_API_KEY)
    if provider == 'aipipe' and Config.AIPIPE_TOKEN: return AIPipeProvider(Config.AIPIPE_TOKEN, model=Config.AIPIPE_MODEL)
    if provider == 'ollama': return OllamaProvider(Config.OLLAMA_BASE_URL, Config.OLLAMA_MODEL)
    raise ValueError("No valid LLM provider configured.")

# --- API Route Definitions ---

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "1.2.1"}

@app.get("/config")
async def get_config_endpoint():
    # FIX: Added missing keys and corrected timeout key for frontend compatibility
    return {
        "llm_provider": Config.DEFAULT_LLM_PROVIDER,
        "model": Config.AIPIPE_MODEL if Config.DEFAULT_LLM_PROVIDER == 'aipipe' else Config.OLLAMA_MODEL,
        "max_file_size_mb": Config.MAX_FILE_SIZE // (1024 * 1024),
        "request_timeout_seconds": Config.REQUEST_TIMEOUT,
        "has_gemini_key": bool(Config.GEMINI_API_KEY),
        "has_aipipe_token": bool(Config.AIPIPE_TOKEN),
        "ollama_model": Config.OLLAMA_MODEL
    }

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """Main endpoint for data analysis"""
    questions_content, data_files = None, []
    for file in files:
        if file.size and file.size > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File {file.filename} exceeds size limit.")
        if file.filename == "questions.txt":
            questions_content = (await file.read()).decode('utf-8')
        else:
            await file.seek(0)
            data_files.append(file)
    if not questions_content:
        raise HTTPException(status_code=400, detail="questions.txt file is required.")

    agent = None
    try:
        llm_provider = get_llm_provider()
        agent = DataAnalystAgent(llm_provider)
        result = await agent.analyze(questions_content, data_files)
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # FIX: Robust cleanup of temporary directory
        if agent and hasattr(agent, 'temp_dir') and os.path.exists(agent.temp_dir):
            try:
                shutil.rmtree(agent.temp_dir)
                logger.info(f"Cleaned up temp directory: {agent.temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temp directory {agent.temp_dir}: {e}")

# --- Static File Mounting ---

static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
if Path("index.html").exists():
    shutil.copy("index.html", static_dir / "index.html")
if (static_dir / "index.html").exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    @app.get("/")
    def root():
        return {"message": "Welcome to the Universal Data Analyst Agent API", "version": "1.2.1"}

# --- Main Execution ---

if __name__ == "__main__":
    print("🚀 Starting Universal Data Analyst Agent v1.2.1...")
    print(f"🔗 Application: http://localhost:8000")
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print(f"💡 Provider: {Config.DEFAULT_LLM_PROVIDER}")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
