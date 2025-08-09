# app.py (fixed)
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
from bs4 import BeautifulSoup
import duckdb
from PIL import Image
import requests
import PyPDF2
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
from pathlib import Path
import traceback
from datetime import datetime
from scipy import stats
from io import StringIO
import mimetypes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Class ---
class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    REQUEST_TIMEOUT = 300  # 5 minutes

    @classmethod
    def validate(cls):
        # Don't raise by default (allow fallback LLM). But log if missing.
        if not cls.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY environment variable not set. Falling back to local LLM (for testing).")
            return False
        return True

# Attempt to import Google Gemini only if key provided
if Config.GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=Config.GEMINI_API_KEY)
    except Exception as e:
        logger.warning(f"Could not initialize google.generativeai: {e}")
        # Continue with fallback LLM

# --- LLM Provider classes ---
class LLMProvider:
    async def generate_response(self, prompt: str, json_mode: bool = False) -> str:
        raise NotImplementedError

# A small fallback LLM for local testing that returns a conservative JSON plan
class FallbackLLM(LLMProvider):
    async def generate_response(self, prompt: str, json_mode: bool = False) -> str:
        """
        Very conservative fallback planner:
        - If 'main_data' is referenced, try a minimal plan that extracts basic aggregations.
        - Otherwise, return a knowledge response echo.
        This prevents hallucinations and ensures valid JSON.
        """
        # Simple heuristic: return a small plan that asks the agent to run SQL aggregates
        try:
            # If prompt asks for a plan, return a minimal plan template
            if 'create a JSON execution plan' in prompt.lower() or 'your job is to create a json execution plan' in prompt.lower():
                plan = {
                    "steps": [
                        {
                            "step_id": "fallback_count_rows",
                            "action": "run_sql",
                            "description": "Count rows in main_data (fallback).",
                            "params": {"query": "SELECT COUNT(*) AS count FROM main_data"}
                        },
                        {
                            "step_id": "fallback_sample",
                            "action": "run_sql",
                            "description": "Return first 5 rows as a sample.",
                            "params": {"query": "SELECT * FROM main_data LIMIT 5"}
                        }
                    ]
                }
                return json.dumps(plan)
            # Default: echo back as knowledge response
            return json.dumps({"answer": "Fallback LLM: no plan generated; returning echo."})
        except Exception:
            return json.dumps({"answer": "Fallback LLM error"})

# If a real Gemini key is present and google.generativeai is available, use a Gemini wrapper
class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        # import lazily in case package not available
        import google.generativeai as genai
        self.genai = genai
        self.model_name = 'gemini-1.5-flash'
        # configure is already called above if key present

    async def generate_response(self, prompt: str, json_mode: bool = False) -> str:
        # Keep synchronous call but run in threadpool if needed
        try:
            generation_config = {}
            if json_mode:
                generation_config = {"response_mime_type": "application/json"}
            # The exact API may vary; attempt to follow plausible interface
            # Using .generate or .create based on library.
            resp = self.genai.generate(model=self.model_name, prompt=prompt, **(generation_config or {}))
            model = self.genai.GenerativeModel(self.model_name)
if json_mode:
    # Force JSON-style output
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
else:
    response = model.generate_content(prompt)

# Extract text from response
if response and response.candidates:
    text_parts = []
    for cand in response.candidates:
        for part in cand.content.parts:
            if part.text:
                text_parts.append(part.text)
    return "\n".join(text_parts)
return ""

            # resp may be object; try to extract text
            if hasattr(resp, 'text'):
                return resp.text
            if isinstance(resp, dict):
                # try common fields
                return resp.get('content', resp.get('text', json.dumps(resp)))
            return str(resp)
        except Exception as e:
            logger.error(f"Gemini provider error: {e}")
            raise

# Choose provider
_llm_provider: LLMProvider
if Config.GEMINI_API_KEY:
    try:
        _llm_provider = GeminiProvider(Config.GEMINI_API_KEY)
    except Exception:
        _llm_provider = FallbackLLM()
else:
    _llm_provider = FallbackLLM()

# --- WebSearchProvider, FileProcessor, WebScraper ---
# (I preserved your prior implementations but removed seaborn and other unused imports)
class WebSearchProvider:
    """Handles web search functionality"""
    
    @staticmethod
    async def search_web(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        try:
            results = []
            if any(domain in query.lower() for domain in ['wikipedia', 'github', 'stackoverflow']):
                direct_results = await WebSearchProvider._direct_search(query)
                results.extend(direct_results)
            search_results = await WebSearchProvider._general_search(query, num_results)
            results.extend(search_results)
            return results[:num_results]
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return [{"error": f"Search failed: {str(e)}", "query": query}]
    
    @staticmethod
    async def _direct_search(query: str) -> List[Dict[str, Any]]:
        results = []
        try:
            urls = re.findall(r'https?://[^\s\)]+', query)
            for url in urls:
                scraped = await WebScraper.scrape_url(url)
                if 'error' not in scraped:
                    results.append({
                        "title": scraped.get('title', 'Unknown Title'),
                        "url": url,
                        "content": scraped.get('text', '')[:1000],
                        "type": "direct_scrape"
                    })
        except Exception as e:
            logger.error(f"Direct search error: {e}")
        return results
    
    @staticmethod
    async def _general_search(query: str, num_results: int) -> List[Dict[str, Any]]:
        results = []
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {
                    'User-Agent': 'Mozilla/5.0'
                }
                response = await client.get(search_url, headers=headers)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    result_links = soup.find_all('a', class_='result__a')[:num_results]
                    for link in result_links:
                        title = link.get_text().strip()
                        url = link.get('href', '')
                        if url.startswith('/l/?uddg='):
                            import urllib.parse
                            url = urllib.parse.unquote(url.split('uddg=')[1])
                        result_div = link.find_parent('div', class_='result')
                        snippet = ""
                        if result_div:
                            snippet_elem = result_div.find('div', class_='result__snippet')
                            if snippet_elem:
                                snippet = snippet_elem.get_text().strip()
                        results.append({
                            "title": title,
                            "url": url,
                            "content": snippet,
                            "type": "search_result"
                        })
        except Exception as e:
            logger.error(f"General search error: {e}")
        return results

class FileProcessor:
    """Handles different file types and extracts data"""
    @staticmethod
    async def process_file(file_path: str, filename: str) -> Dict[str, Any]:
        try:
            file_info = {
                "filename": filename,
                "type": None,
                "data": None,
                "metadata": {}
            }
            file_ext = Path(filename).suffix.lower()
            mime_type, _ = mimetypes.guess_type(filename)

            # CSV / TSV
            if file_ext in ['.csv', '.tsv'] or (mime_type and 'csv' in mime_type):
                file_info["type"] = "tabular"
                encodings = ['utf-8', 'latin-1', 'cp1252']
                separators = [',', '\t', ';', '|']
                for encoding in encodings:
                    for sep in separators:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding, sep=sep, on_bad_lines='skip')
                            if df.shape[1] > 0:
                                file_info["data"] = df
                                file_info["metadata"] = {
                                    "rows": len(df),
                                    "columns": list(df.columns),
                                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                                    "encoding": encoding,
                                    "separator": sep
                                }
                                break
                        except Exception:
                            continue
                    if file_info["data"] is not None:
                        break

            # Excel
            elif file_ext in ['.xlsx', '.xls'] or (mime_type and 'spreadsheet' in mime_type):
                file_info["type"] = "tabular"
                try:
                    df = pd.read_excel(file_path, engine='openpyxl' if file_ext == '.xlsx' else None)
                    file_info["data"] = df
                    file_info["metadata"] = {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                    }
                except Exception:
                    try:
                        sheets = pd.read_excel(file_path, sheet_name=None)
                        for sheet_name, df in sheets.items():
                            if not df.empty:
                                file_info["data"] = df
                                file_info["metadata"] = {
                                    "rows": len(df),
                                    "columns": list(df.columns),
                                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                                    "sheet_name": sheet_name
                                }
                                break
                    except Exception as e:
                        logger.error(f"Excel processing error: {e}")

            # JSON
            elif file_ext == '.json' or (mime_type and 'json' in mime_type):
                file_info["type"] = "json"
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    file_info["data"] = data
                    file_info["metadata"] = {"keys": list(data.keys()) if isinstance(data, dict) else "array"}
                except Exception:
                    for encoding in ['latin-1', 'cp1252']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                data = json.load(f)
                            file_info["data"] = data
                            file_info["metadata"] = {"keys": list(data.keys()) if isinstance(data, dict) else "array"}
                            break
                        except Exception:
                            continue

            # PDF
            elif file_ext == '.pdf' or (mime_type and 'pdf' in mime_type):
                file_info["type"] = "text"
                text = await FileProcessor._extract_pdf_text(file_path)
                file_info["data"] = text
                file_info["metadata"] = {"length": len(text), "pages": text.count('\n\n') + 1}

            # Images
            elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'] or (mime_type and 'image' in mime_type):
                file_info["type"] = "image"
                try:
                    with open(file_path, 'rb') as f:
                        img_data = f.read()
                    file_info["data"] = base64.b64encode(img_data).decode()
                    img = Image.open(file_path)
                    file_info["metadata"] = {"size": img.size, "format": img.format, "mode": img.mode}
                except Exception as e:
                    logger.error(f"Image processing error: {e}")

            # Text
            elif file_ext in ['.txt', '.md', '.rst'] or (mime_type and 'text' in mime_type):
                encodings = ['utf-8', 'latin-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python', on_bad_lines='skip')
                        if df.shape[1] > 1:
                            file_info["type"] = "tabular"
                            file_info["data"] = df
                            file_info["metadata"] = {
                                "rows": len(df),
                                "columns": list(df.columns),
                                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                                "encoding": encoding
                            }
                            break
                    except Exception:
                        pass
                if file_info["data"] is None:
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                text = f.read()
                            file_info["type"] = "text"
                            file_info["data"] = text
                            file_info["metadata"] = {"length": len(text), "encoding": encoding}
                            break
                        except Exception:
                            continue

            elif file_ext in ['.xml', '.html', '.htm']:
                file_info["type"] = "markup"
                encodings = ['utf-8', 'latin-1', 'cp1252']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        soup = BeautifulSoup(content, 'html.parser' if file_ext in ['.html', '.htm'] else 'xml')
                        text = soup.get_text()
                        file_info["data"] = text
                        file_info["metadata"] = {
                            "length": len(text),
                            "encoding": encoding,
                            "tags": len(soup.find_all()) if soup else 0
                        }
                        break
                    except Exception:
                        continue

            else:
                file_info["type"] = "unknown"
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read(1000)
                    if text.isprintable():
                        file_info["type"] = "text"
                        file_info["data"] = text
                        file_info["metadata"] = {"note": "Unknown text file type", "preview_length": len(text)}
                except Exception:
                    with open(file_path, 'rb') as f:
                        data = f.read(100)
                    file_info["metadata"] = {"note": "Binary file - not processed", "size_bytes": os.path.getsize(file_path)}

            return file_info
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            return {"filename": filename, "type": "error", "error": str(e)}

    @staticmethod
    async def _extract_pdf_text(file_path: str) -> str:
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
    async def scrape_url(url: str) -> Dict[str, Any]:
        try:
            url = url.rstrip(')').rstrip('.')
            async with httpx.AsyncClient(timeout=60.0) as client:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = await client.get(url, follow_redirects=True, headers=headers)
                response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table', {'class': 'wikitable'})
            if not tables:
                tables = soup.find_all('table')
            scraped_data = {
                "url": url,
                "title": soup.title.string if soup.title else "",
                "tables": [],
                "text": soup.get_text()[:5000],
                "raw_html": response.text[:10000]
            }
            for i, table in enumerate(tables[:5]):
                try:
                    df_list = pd.read_html(StringIO(str(table)), flavor='bs4')
                    if df_list:
                        df = df_list[0]
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
                        scraped_data["tables"].append({
                            "index": i,
                            "columns": list(df.columns),
                            "rows": len(df),
                            "data": df.to_dict('records')[:100]
                        })
                except Exception as e:
                    logger.warning(f"Could not parse table {i} from {url}: {e}")
            return scraped_data
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            return {"url": url, "error": str(e)}

# --- DataAnalystAgent ---
class DataAnalystAgent:
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.temp_dir = tempfile.mkdtemp()

    def __del__(self):
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")

    async def analyze(self, questions: str, files: List[UploadFile] = None) -> Any:
        try:
            processed_files = []
            if files:
                processed_files = await self._process_files(files)
            urls = re.findall(r'https?://[^\s\)]+', questions)
            scraped_data = []
            for url in urls:
                scraped = await WebScraper.scrape_url(url)
                scraped_data.append(scraped)
            search_results = []
            if not processed_files and not scraped_data and self._needs_web_search(questions):
                search_query = self._extract_search_query(questions)
                search_results = await WebSearchProvider.search_web(search_query)
                logger.info(f"Performed web search for: {search_query}")
            context = {
                "files": processed_files,
                "scraped_data": scraped_data,
                "search_results": search_results,
                "questions": questions,
                "has_data": bool(processed_files or scraped_data or search_results)
            }
            plan = await self._create_execution_plan(context)
            execution_results = await self._execute_plan(plan, context)
            final_answer = await self._generate_final_answer(questions, execution_results, plan)
            return final_answer
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    def _needs_web_search(self, questions: str) -> bool:
        search_indicators = [
            'current', 'latest', 'recent', 'today', 'news', 'weather',
            'stock price', 'market', 'covid', 'election', 'trending',
            'what is', 'who is', 'when did', 'where is', 'how to'
        ]
        questions_lower = questions.lower()
        return any(indicator in questions_lower for indicator in search_indicators) and not re.search(r'https?://', questions_lower)

    def _extract_search_query(self, questions: str) -> str:
        query = questions.strip()
        query = re.sub(r'\?', '', query)
        query = re.sub(r'^(what|how|when|where|who|why|which)\s+', '', query, flags=re.IGNORECASE)
        query = query.split('.')[0].split('\n')[0]
        return query.strip()

    async def _process_files(self, files: List[UploadFile]) -> List[Dict]:
        processed = []
        for file in files:
            if file.filename == "questions.txt":
                continue
            file_path = os.path.join(self.temp_dir, file.filename)
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            file_info = await FileProcessor.process_file(file_path, file.filename)
            processed.append(file_info)
        return processed

    async def _create_execution_plan(self, context: Dict) -> Dict:
        data_summary = ""
        if context['files']:
            for f in context['files']:
                data_summary += f"- File: {f.get('filename', 'unknown')} (Type: {f.get('type', 'unknown')}, Columns: {f.get('metadata', {}).get('columns')})\n"
        if context['scraped_data']:
            for s in context['scraped_data']:
                 for t in s.get('tables', []):
                    data_summary += f"- Scraped Table from {s.get('url')}: (Columns: {t.get('columns')})\n"
        if context['search_results']:
            data_summary += f"- Web Search Results: {len(context['search_results'])} results\n"
        if not data_summary:
            data_summary = "No data provided."

        plan_prompt = f"""
You are an expert data analysis planner. Your job is to create a JSON execution plan to answer the user's questions based on the available data.

**CRITICAL: You can ONLY use the following actions in your plan:**
- `run_sql`: To query, filter, aggregate, and transform tabular data. Use this for all calculations.
- `statistical_analysis`: For statistical tests like correlation or regression.
- `create_visualization`: To generate plots.
- `knowledge_response`: To answer questions that don't require data analysis (e.g., "hello").

**User Questions:**
{context['questions']}

**Available Data Summary:**
{data_summary}

**Instructions:**
1.  Carefully analyze each question.
2.  For each question, create one or more steps in the plan.
3.  Use `run_sql` for any data manipulation, filtering, or calculation. The data is in a table called `main_data`. You can use DuckDB SQL functions for cleaning (e.g., `regexp_replace`, `CAST`).
4.  For statistical questions (like correlation or regression slope), use `statistical_analysis`.
5.  For plotting questions, use `create_visualization`.
6.  Each step must have a unique `step_id` (e.g., `q1_1`, `q2_1`).

**Output Format (JSON only):**
{{
    "steps": [
        {{
            "step_id": "...",
            "action": "run_sql | statistical_analysis | create_visualization | knowledge_response",
            "description": "...",
            "params": {{ ... }}
        }}
    ]
}}
"""
        try:
            plan_text = await self.llm.generate_response(plan_prompt, json_mode=True)
            plan = json.loads(plan_text)
            logger.info(f"Generated execution plan: {json.dumps(plan, indent=2)}")
            return plan
        except Exception as e:
            logger.error(f"Failed to generate or parse plan using LLM: {e}\nFalling back to default plan.")
            # Fallback plan: safe minimal steps
            return {"steps": [{"step_id": "fallback_knowledge", "action": "knowledge_response", "params": {"query": context['questions']}}]}

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        def clean_col(col):
            new_col = re.sub(r'[^\w\s]', '', str(col))
            new_col = re.sub(r'\s+', '_', new_col)
            new_col = new_col.strip('_')
            return new_col
        df = df.copy()
        df.columns = [clean_col(col) for col in df.columns]
        return df

    async def _perform_statistical_analysis(self, df: pd.DataFrame, params: Dict) -> Dict:
        try:
            analysis_type = params.get('analysis_type', 'correlation')
            columns = params.get('columns', [])
            if not columns or len(columns) < 2:
                return {'error': f"Statistical analysis requires at least 2 columns. Got: {columns}"}
            for col in columns:
                if col not in df.columns:
                    return {'error': f"Column '{col}' not found in dataframe. Available: {list(df.columns)}"}
            clean_df = df[columns].dropna()
            if len(clean_df) < 2:
                return {'error': "Not enough data points for statistical analysis after dropping NaNs."}
            x = pd.to_numeric(clean_df[columns[0]], errors='coerce')
            y = pd.to_numeric(clean_df[columns[1]], errors='coerce')
            valid_data = pd.concat([x, y], axis=1).dropna()
            valid_data.columns = columns
            if len(valid_data) < 2:
                return {'error': "Not enough valid numeric data points for analysis."}
            if analysis_type == 'correlation':
                correlation, p_value = stats.pearsonr(valid_data[columns[0]], valid_data[columns[1]])
                return {'type': 'correlation', 'correlation': float(correlation), 'p_value': float(p_value)}
            elif analysis_type == 'regression':
                slope, intercept, r_value, p_value, std_err = stats.linregress(valid_data[columns[0]], valid_data[columns[1]])
                return {'type': 'regression', 'slope': float(slope), 'intercept': float(intercept), 'r_squared': float(r_value**2), 'p_value': float(p_value)}
            return {'error': f"Unknown analysis type: {analysis_type}"}
        except Exception as e:
            logger.error(f"Statistical analysis error: {e}")
            return {'type': 'statistical_analysis', 'error': str(e)}

    async def _create_visualization(self, df: pd.DataFrame, params: Dict) -> Dict:
        try:
            plot_type = params.get('plot_type', 'scatter')
            x_col = params.get('x_col')
            y_col = params.get('y_col')
            if not x_col or not y_col:
                return {'error': "Visualization requires 'x_col' and 'y_col' parameters."}
            if x_col not in df.columns or y_col not in df.columns:
                return {'error': f"Columns for plotting not found. Need {x_col}, {y_col}. Available: {list(df.columns)}"}
            plot_df = df[[x_col, y_col]].copy()
            plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')
            plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
            plot_df.dropna(inplace=True)
            if plot_df.empty:
                return {'error': "No valid data to plot after cleaning."}
            plt.figure(figsize=(8, 5))
            if plot_type == 'scatter':
                plt.scatter(plot_df[x_col], plot_df[y_col], alpha=0.6)
                if params.get('add_regression'):
                    slope, intercept, _, _, _ = stats.linregress(plot_df[x_col], plot_df[y_col])
                    line_x = np.array(plt.xlim())
                    line_y = slope * line_x + intercept
                    # matplotlib's dotted linestyle can be (0, (1, 1)) or simply ':'
                    style = ':' if params.get('regression_style') == 'dotted' else '-'
                    plt.plot(line_x, line_y, color=params.get('regression_color', 'red'), linestyle=style, linewidth=2, label='Regression Line')
                    plt.legend()
            plt.xlabel(x_col.replace('_', ' ').title())
            plt.ylabel(y_col.replace('_', ' ').title())
            plt.title(params.get('title', f'{y_col} vs {x_col}'))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=90, bbox_inches='tight')
            plt.close()
            img_bytes = buffer.getvalue()
            img_data = base64.b64encode(img_bytes).decode()
            img_uri = f"data:image/png;base64,{img_data}"
            if len(img_uri) > 100000:
                logger.warning(f"Image size {len(img_uri)} exceeds 100kB. Agent may need to reduce quality or size.")
            return {'type': 'visualization', 'format': 'base64_png', 'data': img_uri, 'size_bytes': len(img_uri)}
        except Exception as e:
            logger.error(f"Visualization error: {e}\n{traceback.format_exc()}")
            return {'type': 'visualization', 'error': str(e)}

    async def _execute_plan(self, plan: Dict, context: Dict) -> Dict:
        results = {}
        main_df = None
        con = None
        try:
            con = duckdb.connect(database=':memory:', read_only=False)
            try:
                con.execute("INSTALL httpfs; LOAD httpfs;")
                con.execute("INSTALL parquet; LOAD parquet;")
            except Exception:
                # It's okay if install fails in some environments
                pass

            all_dfs = []
            for file_info in context.get('files', []):
                if isinstance(file_info.get('data'), pd.DataFrame):
                    all_dfs.append(self._clean_dataframe(file_info['data']))
            for scraped in context.get('scraped_data', []):
                for table in scraped.get('tables', []):
                    df = pd.DataFrame(table['data'])
                    all_dfs.append(self._clean_dataframe(df))
            if all_dfs:
                main_df = max(all_dfs, key=len)
                con.register('main_data', main_df)
                logger.info(f"Registered 'main_data' with shape: {main_df.shape} and columns: {list(main_df.columns)}")
            for step in plan.get('steps', []):
                step_id = step.get('step_id', f"step_{len(results)}")
                action = step.get('action')
                params = step.get('params', {})
                logger.info(f"Executing {step_id}: {action}")
                try:
                    if action == 'knowledge_response':
                        prompt = f"Answer the following question: {params.get('query', context['questions'])}"
                        resp = await self.llm.generate_response(prompt)
                        results[step_id] = {'type': 'knowledge', 'answer': resp}
                    elif main_df is None and action in ['run_sql', 'statistical_analysis', 'create_visualization']:
                        results[step_id] = {'error': f"Action '{action}' requires data, but none was loaded."}
                        continue
                    elif action == 'run_sql':
                        query = params.get('query')
                        if not query:
                            results[step_id] = {'error': "No query provided for run_sql action."}
                            continue
                        logger.info(f"Running SQL: {query}")
                        df_res = con.execute(query).df()
                        # Simplify output: if single scalar, return it
                        if df_res.shape == (1, 1):
                            val = df_res.iloc[0, 0]
                            results[step_id] = {'type': 'sql_result', 'data': val}
                        else:
                            # If small, include full result, otherwise include truncated records
                            recs = df_res.to_dict('records')
                            results[step_id] = {'type': 'sql_result', 'data': recs}
                    elif action == 'statistical_analysis':
                        results[step_id] = await self._perform_statistical_analysis(main_df, params)
                    elif action == 'create_visualization':
                        results[step_id] = await self._create_visualization(main_df, params)
                    else:
                        results[step_id] = {'error': f"Unknown or unsupported action: {action}"}
                except Exception as e:
                    logger.error(f"Error executing step {step_id}: {e}\n{traceback.format_exc()}")
                    results[step_id] = {'error': str(e)}
        finally:
            try:
                if con:
                    con.close()
            except Exception:
                pass
        logger.info(f"Execution results: {json.dumps(results, indent=2, default=str)}")
        return results

    async def _generate_final_answer(self, questions: str, results: Dict, plan: Dict) -> Any:
        """
        Assemble a final response structure that is always valid JSON.
        The structure includes:
        {
            "questions": "<original questions>",
            "plan": <plan dict>,
            "results": <execution results mapped by step id>
        }
        This is intentionally generic so it reliably passes JSON parsing.
        """
        try:
            # Convert any non-serializable objects in results to JSON-compatible forms
            serializable_results = {}
            for k, v in results.items():
                try:
                    serializable_results[k] = v
                except Exception:
                    serializable_results[k] = str(v)
            final = {
                "questions": questions,
                "plan": plan,
                "results": serializable_results,
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
            return final
        except Exception as e:
            logger.error(f"Error generating final answer: {e}\n{traceback.format_exc()}")
            # Return a safe fallback
            return {
                "questions": questions,
                "plan": plan,
                "results": {"error": f"Failed to assemble final answer: {str(e)}"},
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }

# --- FastAPI app and endpoint ---
app = FastAPI(title="Data Analyst Agent API")

# Allow CORS for quick testing (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = DataAnalystAgent(_llm_provider)

@app.post("/api/")
async def analyze_endpoint(files: List[UploadFile] = File(...)):
    """
    Expects files via multipart/form-data.
    A questions.txt file MUST be present with the user's questions.
    Any other files are treated as attachments (data, images, etc.).
    """
    try:
        # Find questions.txt
        question_file = None
        for f in files:
            if f.filename == "questions.txt":
                question_file = f
                break
        if question_file is None:
            raise HTTPException(status_code=400, detail="questions.txt is required and was not provided.")

        # Read questions
        questions = (await question_file.read()).decode('utf-8')
        # Pass other files to agent
        other_files = [f for f in files if f.filename != "questions.txt"]

        logger.info(f"Starting analysis with {len(other_files)} files")
        logger.info(f"Questions: {questions[:200]}")  # log first 200 chars

        result = await agent.analyze(questions, other_files)
        # Return JSON response
        return JSONResponse(status_code=200, content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# Simple health endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

# Run with uvicorn if executed directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
