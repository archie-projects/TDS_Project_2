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
from fastapi.responses import JSONResponse
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
from urllib.parse import quote
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FIX: Custom JSON Response class to handle NaN values
class NanSafeJSONResponse(JSONResponse):
    """
    Custom JSONResponse class to handle NaN values, converting them to null.
    """
    def render(self, content: Any) -> bytes:
        def convert_nan_to_null(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: convert_nan_to_null(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_nan_to_null(i) for i in obj]
            if isinstance(obj, float) and math.isnan(obj):
                return None
            return obj
        
        cleaned_content = convert_nan_to_null(content)
        return super().render(cleaned_content)

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
        self.token, self.base_url, self.model = token, base_url, model
        self.client = httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT)
    async def generate_response(self, prompt: str, json_mode: bool = False) -> str:
        try:
            payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
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
        self.base_url, self.model = base_url, model
        self.client = httpx.AsyncClient(timeout=Config.REQUEST_TIMEOUT)
    async def generate_response(self, prompt: str, json_mode: bool = False) -> str:
        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            if json_mode:
                payload["format"] = "json"
                prompt += "\n\nIMPORTANT: Respond with valid JSON only, no other text."
            response = await self.client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

class FileProcessor:
    @staticmethod
    def _extract_pdf_text(file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    if page_text := page.extract_text():
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
        return text

    @staticmethod
    async def process_file(file_path: str, filename: str) -> Dict[str, Any]:
        try:
            file_info = {"filename": filename, "type": "unknown", "data": None, "metadata": {}}
            file_ext = Path(filename).suffix.lower()
            if file_ext in ['.csv', '.tsv']:
                file_info.update({"type": "tabular", "data": pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')})
            elif file_ext in ['.xlsx', '.xls']:
                file_info.update({"type": "tabular", "data": pd.read_excel(file_path)})
            elif file_ext == '.json':
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    file_info.update({"type": "json", "data": json.loads(await f.read())})
            elif file_ext == '.pdf':
                file_info.update({"type": "text", "data": await asyncio.to_thread(FileProcessor._extract_pdf_text, file_path)})
            elif file_ext in ['.png', '.jpg', '.jpeg']:
                async with aiofiles.open(file_path, 'rb') as f:
                    img_data = await f.read()
                img = Image.open(file_path)
                file_info.update({"type": "image", "data": base64.b64encode(img_data).decode(), "metadata": {"size": img.size, "format": img.format}})
            elif file_ext == '.txt':
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', sep=r'\s{2,}', engine='python')
                    if df.shape[1] > 1: file_info.update({"type": "tabular", "data": df})
                    else: raise ValueError()
                except Exception:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        file_info.update({"type": "text", "data": await f.read()})
            if file_info["type"] == "tabular":
                df = file_info["data"]
                file_info["metadata"] = {"rows": len(df), "columns": list(df.columns), "dtypes": {c: str(t) for c, t in df.dtypes.items()}}
            elif file_info["type"] == "text":
                file_info["metadata"] = {"length": len(file_info.get("data", ""))}
            elif file_info["type"] == "json":
                data = file_info["data"]
                file_info["metadata"] = {"keys": list(data.keys()) if isinstance(data, dict) else "array"}
            return file_info
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}\n{traceback.format_exc()}")
            return {"filename": filename, "type": "error", "error": str(e)}

class WebScraper:
    @staticmethod
    def _clean_url(url: str) -> str:
        url = url.strip().rstrip('.,)]}').strip().replace('`', '')
        if '://' in url:
            scheme, rest = url.split('://', 1)
            if '/' in rest:
                domain, path = rest.split('/', 1)
                return f"{scheme}://{domain}/{quote(path, safe='/:?#[]@!$&()*+,;=')}"
        return url

    @staticmethod
    async def scrape_url(url: str, proxy: bool = True) -> Dict[str, Any]:
        try:
            clean_url = WebScraper._clean_url(url)
            scrape_url = f"https://aipipe.org/proxy/{clean_url}" if proxy else clean_url
            logger.info(f"Scraping URL: {clean_url}")
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(scrape_url)
                response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = []
            for i, table in enumerate(soup.find_all('table', limit=5)):
                try:
                    df = pd.read_html(StringIO(str(table)))[0]
                    tables.append({"index": i, "columns": list(df.columns), "data": df.to_dict('records')})
                except Exception as e:
                    logger.warning(f"Could not parse table {i} from {clean_url}: {e}")
            return {"url": clean_url, "title": soup.title.string if soup.title else "", "tables": tables, "text": soup.get_text(separator='\n', strip=True)[:5000]}
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            if proxy: return await WebScraper.scrape_url(url, proxy=False)
            return {"url": url, "error": str(e)}

class DataAnalystAgent:
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.temp_dir = tempfile.mkdtemp()
    
    async def analyze(self, questions: str, files: List[UploadFile]) -> Any:
        processed_files, scraped_data = await asyncio.gather(
            self._process_files(files),
            asyncio.gather(*(WebScraper.scrape_url(url) for url in re.findall(r'https?://[^\s()<>]+', questions)))
        )
        context = {"files": processed_files, "scraped_data": scraped_data, "questions": questions}
        plan = await self._create_execution_plan(context)
        execution_results = await self._execute_plan(plan, context)
        return await self._generate_final_answer(questions, execution_results, context)

    async def _process_files(self, files: List[UploadFile]) -> List[Dict]:
        tasks = [self._process_single_file(f) for f in files if f.filename != "questions.txt"]
        return await asyncio.gather(*tasks)

    async def _process_single_file(self, file: UploadFile) -> Dict:
        file_path = os.path.join(self.temp_dir, file.filename)
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(await file.read())
        return await FileProcessor.process_file(file_path, file.filename)

    async def _create_execution_plan(self, context: Dict) -> Dict:
        simple_context = {
            'files': [{'name': f.get('filename'), 'type': f.get('type'), 'metadata': f.get('metadata')} for f in context['files']],
            'scraped_data': [{'url': s.get('url'), 'title': s.get('title', ''), 'tables_found': len(s.get('tables', [])) > 0} for s in context['scraped_data'] if 'error' not in s]
        }
        plan_prompt = f"Create a JSON execution plan to answer these questions:\n{context['questions']}\n\nBased on this data context:\n{json.dumps(simple_context, indent=2)}\n\nActions can be 'run_sql', 'statistical_analysis', 'create_visualization'. For SQL, use valid syntax (e.g., `WHERE amount > 500`, not `500_crore`). Respond with JSON plan only."
        try:
            plan_text = await self.llm.generate_response(plan_prompt, json_mode=True)
            plan = json.loads(re.sub(r'^```json\s*|\s*```$', '', plan_text, flags=re.I))
            logger.info(f"Generated plan: {plan}")
            return plan
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            return {"steps": []}

    async def _execute_plan(self, plan: Dict, context: Dict) -> Dict:
        results, main_df = {}, None
        con = duckdb.connect(database=':memory:')
        try:
            for i, f in enumerate(context.get('files', [])):
                if f.get('type') == 'tabular' and isinstance(f.get('data'), pd.DataFrame):
                    df = self._clean_dataframe(f['data'])
                    con.register(f"tbl_{i}", df)
                    if main_df is None: main_df = df; con.register('main_data', df)
            for i, s in enumerate(context.get('scraped_data', [])):
                if 'error' not in s:
                    for j, t in enumerate(s.get('tables', [])):
                        df = self._clean_dataframe(pd.DataFrame(t['data']))
                        con.register(f"scraped_{i}_{j}", df)
                        if main_df is None: main_df = df; con.register('main_data', df)
            for step in plan.get('steps', []):
                sid, action, params = step['step_id'], step['action'], step.get('params', {})
                logger.info(f"Executing step {sid}: {action}")
                try:
                    if action == 'run_sql' and params.get('query'):
                        query = re.sub(r'(\d+)_crore', r'\1', params['query'])
                        results[sid] = {'type': 'dataframe', 'data': con.execute(query).df().to_dict('records')}
                    elif main_df is not None:
                        if action == 'statistical_analysis':
                            results[sid] = await self._perform_statistical_analysis(main_df.copy(), params)
                        elif action == 'create_visualization':
                            results[sid] = await self._create_visualization(main_df.copy(), params)
                except Exception as e:
                    results[sid] = {'type': 'error', 'error': str(e)}
        finally:
            con.close()
        return results

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(c)).strip('_') for c in df.columns]
        for col in df.columns:
            if df[col].dtype == 'object':
                if 'date' in col.lower():
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif df[col].astype(str).str.contains(r'[\d,₹]+', na=False).any():
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[₹,]', '', regex=True), errors='coerce')
        return df

    async def _perform_statistical_analysis(self, df: pd.DataFrame, params: Dict) -> Dict:
        try:
            return {'type': 'summary', 'data': df.describe()}
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

    async def _create_visualization(self, df: pd.DataFrame, params: Dict) -> Dict:
        try:
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) < 2: return {'type': 'error', 'message': 'Not enough numeric columns.'}
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.7)
            ax.set_xlabel(numeric_cols[0]); ax.set_ylabel(numeric_cols[1])
            ax.set_title(f"Scatter Plot of {numeric_cols[1]} vs {numeric_cols[0]}")
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png'); plt.close(fig)
            return {'type': 'visualization', 'data': f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"}
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

    async def _generate_final_answer(self, questions: str, results: Dict, context: Dict) -> Any:
        summary = {f"step_{sid}": res for sid, res in results.items()}
        prompt = f"Answer these questions:\n{questions}\n\nUsing these results:\n{json.dumps(summary, indent=2, default=str)}\n\nFormat your response as a single JSON object."
        try:
            response_text = await self.llm.generate_response(prompt, json_mode=True)
            return json.loads(re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.I))
        except Exception as e:
            logger.error(f"Final answer synthesis failed: {e}")
            return {"error": "Failed to generate final answer", "raw_results": summary}

# --- FastAPI App ---
app = FastAPI(title="Universal Data Analyst Agent", version="1.2.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
Config.validate()

def get_llm_provider():
    p_type = Config.DEFAULT_LLM_PROVIDER
    if p_type == 'gemini' and Config.GEMINI_API_KEY: return GeminiProvider(Config.GEMINI_API_KEY)
    if p_type == 'aipipe' and Config.AIPIPE_TOKEN: return AIPipeProvider(Config.AIPIPE_TOKEN, model=Config.AIPIPE_MODEL)
    if p_type == 'ollama': return OllamaProvider(Config.OLLAMA_BASE_URL, Config.OLLAMA_MODEL)
    raise ValueError("No valid LLM provider configured.")

@app.get("/health")
async def health_check(): return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/config")
async def get_config_endpoint():
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
    q_content, data_files = None, []
    for f in files:
        if f.size and f.size > Config.MAX_FILE_SIZE:
            raise HTTPException(413, "File size limit exceeded.")
        if f.filename == "questions.txt":
            q_content = (await f.read()).decode()
        else:
            await f.seek(0); data_files.append(f)
    if not q_content: raise HTTPException(400, "questions.txt is required.")
    
    agent = None
    try:
        agent = DataAnalystAgent(get_llm_provider())
        result = await agent.analyze(q_content, data_files)
        # FIX: Use the NanSafeJSONResponse to handle NaN values
        return NanSafeJSONResponse(content=result)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")
    finally:
        if agent and os.path.exists(agent.temp_dir):
            shutil.rmtree(agent.temp_dir)

# Static file serving
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
if (index_path := Path("index.html")).exists():
    shutil.copy(index_path, static_dir / index_path.name)
if (static_dir / "index.html").exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    @app.get("/")
    def root(): return {"message": "Welcome to the Data Analyst Agent API. No UI found."}

if __name__ == "__main__":
    print(f"🚀 Starting Server v{app.version}... | http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
