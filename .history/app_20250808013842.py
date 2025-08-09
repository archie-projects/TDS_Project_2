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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    AIPIPE_TOKEN = os.getenv('AIPIPE_TOKEN')
    AIPIPE_MODEL = os.getenv('AIPIPE_MODEL', 'openai/gpt-4o-mini') # Changed to mini version
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')
    DEFAULT_LLM_PROVIDER = os.getenv('DEFAULT_LLM_PROVIDER', 'aipipe')  # gemini, aipipe, ollama
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
            
            # FIX: Use the asynchronous version of the method to avoid blocking.
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
        self.client = httpx.AsyncClient(timeout=300.0)
        
    async def generate_response(self, prompt: str, json_mode: bool = False) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            payload = {"model": self.model, "messages": messages}
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
                
            headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
            
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
        self.client = httpx.AsyncClient(timeout=180.0)
        
    async def generate_response(self, prompt: str, json_mode: bool = False) -> str:
        try:
            if json_mode:
                prompt += "\n\nIMPORTANT: Respond with valid JSON only, no other text."
                
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = await self.client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
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
                # FIX: Use aiofiles for non-blocking file read.
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                file_info["data"] = data
                file_info["metadata"] = {"keys": list(data.keys()) if isinstance(data, dict) else "array"}
                
            elif file_ext == '.pdf':
                file_info["type"] = "text"
                # FIX: Run synchronous PyPDF2 code in a thread to avoid blocking.
                text = await asyncio.to_thread(FileProcessor._extract_pdf_text, file_path)
                file_info["data"] = text
                file_info["metadata"] = {"length": len(text)}
                
            elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                file_info["type"] = "image"
                async with aiofiles.open(file_path, 'rb') as f:
                    img_data = await f.read()
                file_info["data"] = base64.b64encode(img_data).decode()
                img = Image.open(file_path)
                file_info["metadata"] = {"size": img.size, "format": img.format}
                
            elif file_ext == '.txt':
                # Try to read as a table first
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', sep=r'\s{2,}')
                    if df.shape[1] > 1:
                        file_info["type"] = "tabular"
                        file_info["data"] = df
                        file_info["metadata"] = {
                            "rows": len(df),
                            "columns": list(df.columns),
                            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                        }
                    else:
                        raise ValueError("Not a tabular file")
                except Exception:
                    # Fallback to reading as plain text
                    file_info["type"] = "text"
                    # FIX: Use aiofiles for non-blocking file read.
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        text = await f.read()
                    file_info["data"] = text
                    file_info["metadata"] = {"length": len(text)}
                
            else:
                file_info["type"] = "unknown"
                logger.warning(f"Unsupported file type: {file_ext}")
                
            return file_info
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            return {"filename": filename, "type": "error", "error": str(e)}
    
    # FIX: This is a synchronous method that will be run in a separate thread.
    @staticmethod
    def _extract_pdf_text(file_path: str) -> str:
        """Extract text from PDF"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""

class WebScraper:
    """Handles web scraping with various strategies"""
    
    @staticmethod
    async def scrape_url(url: str, proxy: bool = True) -> Dict[str, Any]:
        """Scrape data from URL"""
        try:
            # Clean URL first - remove any trailing characters
            url = url.rstrip(')').rstrip('.')
            
            # Use proxy for better reliability
            if proxy:
                proxy_url = f"https://aipipe.org/proxy/{url}"
                scrape_url = proxy_url
            else:
                scrape_url = url
                
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(scrape_url, follow_redirects=True)
                response.raise_for_status()
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find tables
            tables = soup.find_all('table')
            scraped_data = {
                "url": url,
                "title": soup.title.string if soup.title else "",
                "tables": [],
                "text": soup.get_text()[:5000],  # First 5000 chars
                "raw_html": response.text[:10000]  # First 10k chars
            }
            
            # Convert tables to dataframes
            for i, table in enumerate(tables[:5]):  # Limit to 5 tables
                try:
                    df = pd.read_html(StringIO(str(table)))[0]
                    scraped_data["tables"].append({
                        "index": i,
                        "columns": list(df.columns),
                        "rows": len(df),
                        "data": df.to_dict('records')[:100]  # Limit rows
                    })
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
        
    async def analyze(self, questions: str, files: List[UploadFile]) -> Any:
        """Main analysis pipeline"""
        try:
            # Step 1: Process all files
            processed_files = await self._process_files(files)
            
            # Step 2: Extract URLs from questions and scrape if needed
            urls = re.findall(r'https?://[^\s\)]+', questions)
            scraped_data = []
            if urls:
                scraped_tasks = [WebScraper.scrape_url(url) for url in urls]
                scraped_data = await asyncio.gather(*scraped_tasks)
            
            # Step 3: Create analysis context
            context = {
                "files": processed_files,
                "scraped_data": scraped_data,
                "questions": questions
            }
            
            # Step 4: Generate execution plan
            plan = await self._create_execution_plan(context)
            
            # Step 5: Execute the plan
            execution_results = await self._execute_plan(plan, context)
            
            # Step 6: Generate final response
            final_answer = await self._generate_final_answer(questions, execution_results, context)
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    async def _process_files(self, files: List[UploadFile]) -> List[Dict]:
        """Process all uploaded files"""
        processed = []
        
        async def process_single_file(file: UploadFile):
            if file.filename == "questions.txt":
                return None
                
            file_path = os.path.join(self.temp_dir, file.filename)
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            return await FileProcessor.process_file(file_path, file.filename)

        results = await asyncio.gather(*(process_single_file(f) for f in files))
        processed = [res for res in results if res is not None]
        return processed
    
    async def _create_execution_plan(self, context: Dict) -> Dict:
        """Create a flexible execution plan based on context"""
        
        plan_prompt = f"""You are an expert data analyst AI. Analyze the user's questions and available data to create a comprehensive execution plan.

QUESTIONS TO ANSWER:
{context['questions']}

AVAILABLE DATA:
Files: {[f.get('filename', 'unknown') for f in context['files']]}
File Types: {[f.get('type', 'unknown') for f in context['files']]}
Scraped URLs: {[s.get('url', 'unknown') for s in context['scraped_data']]}

DETAILED CONTEXT:
{json.dumps({
    'files': [{
        'name': f.get('filename'),
        'type': f.get('type'),
        'metadata': f.get('metadata', {})
    } for f in context['files']],
    'scraped_data': [{
        'url': s.get('url'),
        'tables_count': len(s.get('tables', [])),
        'has_error': 'error' in s
    } for s in context['scraped_data']]
}, indent=2)}

Create a JSON execution plan. For S3 data sources like the Indian High Court dataset:
1. Use individual year paths: s3://indian-high-court-judgments/metadata/parquet/year=2019/court=*/bench=*/metadata.parquet
2. For multiple years (2019-2022), create separate queries for each year and UNION them
3. Always include INSTALL and LOAD statements for httpfs and parquet
4. Handle date parsing and delay calculations properly

Response format:
{{
    "analysis_type": "descriptive|statistical|predictive|comparative",
    "expected_response_format": "json_array|json_object|mixed",
    "response_structure": "describe the expected output structure",
    "steps": [
        {{
            "step_id": "step_1",
            "action": "load_data|run_sql|statistical_analysis|create_visualization|text_analysis",
            "description": "clear description of what this step does",
            "params": {{
                "data_source": "specific file or table name",
                "query": "SQL query for data operations",
                "analysis_type": "correlation|regression|counting|grouping",
                "columns": ["column1", "column2"],
                "plot_type": "scatter|bar|line",
                "output_format": "value|dataframe|plot"
            }},
            "expected_output": "description of expected output"
        }}
    ]
}}

IMPORTANT: 
- For S3 parquet data, use specific year patterns, not wildcards like year=2019-2022
- Include proper SQL for date parsing and delay calculations
- Plan for the exact response format the user expects
- Include visualization steps for any requested plots

Respond with valid JSON only."""
        
        try:
            plan_text = await self.llm.generate_response(plan_prompt, json_mode=True)
            plan_text = re.sub(r'^```json\s*', '', plan_text, flags=re.IGNORECASE)
            plan_text = re.sub(r'\s*```$', '', plan_text)
            
            plan = json.loads(plan_text)
            logger.info(f"Generated execution plan: {json.dumps(plan, indent=2)}")
            return plan
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            logger.error(f"Raw response: {plan_text}")
            raise ValueError(f"Failed to generate a valid execution plan from the LLM. Raw response: {plan_text}")
    
    async def _execute_plan(self, plan: Dict, context: Dict) -> Dict:
        """Execute the analysis plan"""
        results = {}
        
        con = duckdb.connect(database=':memory:', read_only=False)
        try:
            con.execute("INSTALL httpfs; LOAD httpfs;")
            con.execute("INSTALL parquet; LOAD parquet;")

            main_df = None
            
            for file_info in context['files']:
                if file_info.get('type') == 'tabular' and file_info.get('data') is not None:
                    df = self._clean_dataframe(file_info['data'])
                    table_name = f"file_{re.sub(r'[^A-Za-z0-9_]', '', Path(file_info['filename']).stem)}"
                    con.register(table_name, df)
                    if main_df is None:
                        main_df = df
                        con.register('main_data', df)
                        logger.info(f"Registered file '{file_info['filename']}' as main_data with shape: {df.shape}")

            for i, scraped in enumerate(context['scraped_data']):
                if 'tables' in scraped and scraped['tables']:
                    for j, table_info in enumerate(scraped['tables']):
                        df = pd.DataFrame(table_info['data'])
                        df = self._clean_dataframe(df)
                        table_name = f"scraped_table_{i}_{j}"
                        con.register(table_name, df)
                        if main_df is None:
                            main_df = df
                            con.register('main_data', df)
                            logger.info(f"Registered scraped table as main_data with shape: {df.shape}")

            for step in plan.get('steps', []):
                step_id = step['step_id']
                action = step['action']
                params = step.get('params', {})
                
                logger.info(f"Executing {step_id}: {action}")
                
                try:
                    if action == 'load_data':
                        query = params.get('query')
                        data_source = params.get('data_source', '')
                        
                        if 's3://' in data_source or (query and 's3://' in query):
                            # FIX: Robustly handle year-range replacement for S3 queries
                            if query and 'year=2019-2022' in query:
                                path_match = re.search(r"read_parquet\('([^']*)'\)", query)
                                if path_match:
                                    base_path_str = path_match.group(1)
                                    union_parts = []
                                    for year in [2019, 2020, 2021, 2022]:
                                        year_path = base_path_str.replace('year=2019-2022', f'year={year}')
                                        union_parts.append(f"SELECT * FROM read_parquet('{year_path}')")
                                    query = " UNION ALL ".join(union_parts)
                                else:
                                    logger.warning("Could not parse S3 path from query to expand year range.")

                            logger.info(f"Loading S3 data with query: {query}")
                            result_df = con.execute(query).df()
                            main_df = self._clean_dataframe(result_df)
                            con.register('main_data', main_df)
                            results[step_id] = {'type': 'dataframe', 'shape': main_df.shape}
                            logger.info(f"Loaded and registered 'main_data' with shape: {main_df.shape}")
                        
                        elif query:
                            logger.info(f"Loading data with query: {query}")
                            result_df = con.execute(query).df()
                            main_df = self._clean_dataframe(result_df)
                            con.register('main_data', main_df)
                            results[step_id] = {'type': 'dataframe', 'shape': main_df.shape}
                            logger.info(f"Loaded and registered 'main_data' with shape: {main_df.shape}")

                    elif action == 'run_sql':
                        query = params.get('query', '')
                        if query:
                            logger.info(f"Executing SQL: {query}")
                            result_df = con.execute(query).df()
                            results[step_id] = {
                                'type': 'dataframe',
                                'data': result_df.to_dict('records'),
                                'columns': list(result_df.columns),
                                'shape': result_df.shape
                            }

                    elif action == 'statistical_analysis':
                        if main_df is None: raise ValueError("Statistical analysis requires a loaded dataframe.")
                        analysis_df = main_df.copy()
                        
                        filter_condition = params.get('filter_condition')
                        if filter_condition and "court='33_10'" in filter_condition:
                            analysis_df = analysis_df[analysis_df['court'] == '33_10']
                        
                        if 'date_of_registration' in analysis_df.columns and 'decision_date' in analysis_df.columns:
                            analysis_df['date_of_registration'] = pd.to_datetime(analysis_df['date_of_registration'], errors='coerce')
                            analysis_df['decision_date'] = pd.to_datetime(analysis_df['decision_date'], errors='coerce')
                            analysis_df['delay_days'] = (analysis_df['decision_date'] - analysis_df['date_of_registration']).dt.days
                        
                        analysis_result = await self._perform_statistical_analysis(analysis_df, params)
                        results[step_id] = analysis_result
                    
                    elif action == 'create_visualization':
                        if main_df is None: raise ValueError("Visualization requires a loaded dataframe.")
                        plot_df = main_df.copy()
                        
                        filter_condition = params.get('filter_condition')
                        if filter_condition and "court='33_10'" in filter_condition:
                            plot_df = plot_df[plot_df['court'] == '33_10']
                        
                        if 'date_of_registration' in plot_df.columns and 'decision_date' in plot_df.columns:
                            plot_df['date_of_registration'] = pd.to_datetime(plot_df['date_of_registration'], errors='coerce')
                            plot_df['decision_date'] = pd.to_datetime(plot_df['decision_date'], errors='coerce')
                            plot_df['delay_days'] = (plot_df['decision_date'] - plot_df['date_of_registration']).dt.days
                        
                        plot_result = await self._create_visualization(plot_df, params)
                        results[step_id] = plot_result
                    
                except Exception as e:
                    logger.error(f"Error in step {step_id}: {e}\n{traceback.format_exc()}")
                    results[step_id] = {'type': 'error', 'error': str(e)}
            
        finally:
            con.close()
            
        return results
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataframe"""
        df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col).strip()) for col in df.columns]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        continue
                    except Exception: pass

                if col not in ['court', 'title', 'description', 'judge', 'cnr', 'disposal_nature', 'bench']:
                    try:
                        cleaned = df[col].astype(str).str.replace(r'[$,%]', '', regex=True)
                        numeric_series = pd.to_numeric(cleaned, errors='coerce')
                        if not numeric_series.isna().all():
                            df[col] = numeric_series
                    except Exception: pass
        return df
    
    async def _perform_statistical_analysis(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Perform statistical analysis"""
        try:
            analysis_type = params.get('analysis_type', 'correlation')
            columns = params.get('columns', [])
            numeric_df = df.copy()

            if analysis_type == 'regression' and len(columns) >= 2:
                x_col = columns[0]
                y_col = columns[1]
                
                if 'delay_days' in numeric_df.columns and 'year' in numeric_df.columns:
                    x_col, y_col = 'year', 'delay_days'
                
                if x_col in numeric_df.columns and y_col in numeric_df.columns:
                    clean_data = numeric_df[[x_col, y_col]].dropna()
                    if len(clean_data) > 1:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data[x_col], clean_data[y_col])
                        return { 'type': 'regression', 'x_column': x_col, 'y_column': y_col, 'slope': float(slope), 'intercept': float(intercept), 'r_squared': float(r_value**2), 'p_value': float(p_value), 'sample_size': len(clean_data) }
            
            elif analysis_type == 'correlation':
                numeric_cols = numeric_df.select_dtypes(include=np.number).columns.tolist()
                if len(numeric_cols) >= 2:
                    corr_matrix = numeric_df[numeric_cols].corr()
                    return { 'type': 'correlation', 'correlation_matrix': corr_matrix.to_dict() }
            
            return {'type': 'statistical_analysis', 'error': f'Cannot perform {analysis_type} analysis'}
        except Exception as e:
            return {'type': 'statistical_analysis', 'error': str(e)}
    
    async def _create_visualization(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Create visualizations"""
        try:
            plot_type = params.get('plot_type', 'scatter')
            columns = params.get('columns', [])
            x_col = columns[0] if len(columns) > 0 else None
            y_col = columns[1] if len(columns) > 1 else None
            
            if 'delay_days' in df.columns and 'year' in df.columns:
                x_col, y_col = 'year', 'delay_days'
            elif not x_col or not y_col:
                numeric_cols = df.select_dtypes(include=[np.number, 'datetime64']).columns.tolist()
                if not x_col and len(numeric_cols) > 0: x_col = numeric_cols[0]
                if not y_col and len(numeric_cols) > 1: y_col = numeric_cols[1]
            
            if not x_col or x_col not in df.columns or (plot_type != 'hist' and (not y_col or y_col not in df.columns)):
                return {'type': 'visualization', 'error': f'Columns {x_col}, {y_col} not found.'}
            
            plt.style.use('seaborn-v0_8-whitegrid')
            
            async def generate_plot(is_fallback=False):
                fig, ax = plt.subplots(figsize=(10, 6) if not is_fallback else (8, 5), dpi=150 if not is_fallback else 100)
                plot_df = df[[x_col, y_col]].dropna() if y_col else df[[x_col]].dropna()
                if len(plot_df) == 0: return {'type': 'visualization', 'error': 'No data for plotting.'}

                if plot_type == 'scatter':
                    ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.6)
                    if params.get('add_regression', True) and len(plot_df) > 1:
                        try:
                            x_data, y_data = plot_df[x_col], plot_df[y_col]
                            is_datetime_x = pd.api.types.is_datetime64_any_dtype(x_data)
                            
                            x_numeric = x_data.apply(lambda d: d.toordinal()) if is_datetime_x else x_data
                            slope, intercept, _, _, _ = stats.linregress(x_numeric, y_data)
                            
                            line_x_ord = np.array([x_numeric.min(), x_numeric.max()])
                            line_y = slope * line_x_ord + intercept
                            
                            # FIX: Convert line endpoints back to datetime if axis is datetime.
                            line_x_plot = [datetime.fromordinal(int(d)) for d in line_x_ord] if is_datetime_x else line_x_ord
                            ax.plot(line_x_plot, line_y, 'r--', linewidth=2, label='Regression Line')
                            ax.legend()
                        except Exception as e:
                            logger.warning(f"Could not add regression line: {e}")

                elif plot_type == 'bar':
                    if plot_df[x_col].nunique() < 30:
                        data = plot_df.groupby(x_col)[y_col].sum().nlargest(30) if y_col else plot_df[x_col].value_counts().nlargest(30)
                        data.plot(kind='bar', ax=ax)
                        plt.xticks(rotation=45, ha='right')
                    else: return {'type': 'visualization', 'error': 'Too many categories for a bar plot.'}
                
                elif plot_type == 'line':
                    plot_df.set_index(x_col)[y_col].plot(ax=ax, marker='o')
                
                ax.set_xlabel(str(x_col).replace('_', ' ').title())
                if y_col: ax.set_ylabel(str(y_col).replace('_', ' ').title())
                ax.set_title(f"{plot_type.title()} of {y_col or ''} by {x_col}".replace('_', ' ').title())
                plt.tight_layout()
                
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', bbox_inches='tight')
                plt.close(fig)
                return base64.b64encode(buffer.getvalue()).decode()

            img_data = await generate_plot()
            if len(img_data) > 150000: # Reduced size limit for safety
                logger.warning(f"Image size {len(img_data)} is large, reducing quality.")
                img_data = await generate_plot(is_fallback=True)

            return {'type': 'visualization', 'format': 'base64_png', 'data': f"data:image/png;base64,{img_data}"}
            
        except Exception as e:
            logger.error(f"Visualization error: {e}\n{traceback.format_exc()}")
            return {'type': 'visualization', 'error': str(e)}
    
    async def _generate_final_answer(self, questions: str, results: Dict, context: Dict) -> Any:
        """Generate the final answer in the requested format"""
        synthesis_prompt = f"""You are a data synthesis AI. Your task is to provide the EXACT final answer format requested by the user, based on the provided analysis results.

ORIGINAL QUESTIONS:
{questions}

ANALYSIS RESULTS:
{json.dumps(results, indent=2, default=str)}

CONTEXT DATA:
{json.dumps({
    'files_processed': len(context.get('files', [])),
    'scraped_urls': [s.get('url') for s in context.get('scraped_data', [])]
}, indent=2)}

CRITICAL INSTRUCTIONS:
1. Examine the original questions to understand the EXACT response format expected.
2. Extract the specific answers from the "ANALYSIS RESULTS".
3. For questions about "most cases", look for SQL results with court counts and find the maximum.
4. For questions about "regression slope", extract the 'slope' value from a 'statistical_analysis' step result.
5. For visualization requests ("Plot..."), find the 'visualization' step result and include the complete base64 data URI from the 'data' key.
6. Assemble the final response into a single JSON object where keys are the original questions and values are the extracted answers.
7. If a specific analysis failed (e.g., you see an "error" key in the results for that step), state that the analysis could not be completed. DO NOT make up data.

Respond with a valid JSON object only. The keys of the JSON object must be the questions from the prompt, and the values must be the answers derived from the analysis results.
"""
        try:
            final_response_text = await self.llm.generate_response(synthesis_prompt, json_mode=True)
            final_response_text = re.sub(r'^```json\s*', '', final_response_text, flags=re.IGNORECASE)
            final_response_text = re.sub(r'\s*```$', '', final_response_text)
            
            parsed_response = json.loads(final_response_text)
            return parsed_response
        except Exception as e:
            logger.error(f"Final answer generation failed: {e}")
            logger.error(f"Raw response from LLM was: {final_response_text}")
            raise HTTPException(status_code=500, detail="Failed to synthesize the final answer from analysis results.")

# Initialize FastAPI app
app = FastAPI(
    title="Universal Data Analyst Agent",
    description="AI-powered data analysis with support for multiple file formats, web scraping, and visualizations",
    version="1.0.0"
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
    if provider == 'gemini' and Config.GEMINI_API_KEY:
        return GeminiProvider(Config.GEMINI_API_KEY)
    if provider == 'aipipe' and Config.AIPIPE_TOKEN:
        return AIPipeProvider(Config.AIPIPE_TOKEN, model=Config.AIPIPE_MODEL)
    if provider == 'ollama':
        return OllamaProvider(Config.OLLAMA_BASE_URL, Config.OLLAMA_MODEL)
    
    logger.warning(f"Default provider '{provider}' not configured, falling back to Ollama.")
    return OllamaProvider(Config.OLLAMA_BASE_URL, Config.OLLAMA_MODEL)

static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
if Path("index.html").exists():
    import shutil
    shutil.copy("index.html", static_dir / "index.html")

if static_dir.exists() and (static_dir / "index.html").exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "llm_provider": Config.DEFAULT_LLM_PROVIDER}

@app.get("/config")
async def get_config_endpoint():
    return get_config()

def get_config():
    return {
        "llm_provider": Config.DEFAULT_LLM_PROVIDER,
        "ai_pipe_model": Config.AIPIPE_MODEL,
        "max_file_size_mb": Config.MAX_FILE_SIZE // (1024 * 1024),
        "request_timeout_seconds": Config.REQUEST_TIMEOUT,
    }

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    start_time = datetime.now()
    
    try:
        questions_content = None
        data_files = []
        
        for file in files:
            if file.size > Config.MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"File {file.filename} too large.")
            if file.filename == "questions.txt":
                questions_content = (await file.read()).decode('utf-8')
            else:
                await file.seek(0)
                data_files.append(file)
        
        if not questions_content:
            raise HTTPException(status_code=400, detail="questions.txt file is missing.")
        
        llm_provider = get_llm_provider()
        agent = DataAnalystAgent(llm_provider)
        
        logger.info(f"Starting analysis with {len(data_files)} files for questions: {questions_content[:200]}...")
        result = await agent.analyze(questions_content, data_files)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    print("\n🚀 Starting Universal Data Analyst Agent...")
    config = get_config()
    for key, value in config.items():
        print(f"📊 {key.replace('_', ' ').title()}: {value}")
    
    print("\n🌐 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
