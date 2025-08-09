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
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

class AIPipeProvider(LLMProvider):
    def __init__(self, token: str, base_url: str = "https://aipipe.org/openrouter/v1", model: str = "openai/gpt-4o-mini"):
        self.token = token
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0))
        
    async def generate_response(self, prompt: str, json_mode: bool = False) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            payload = {"model": self.model, "messages": messages}
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
                
            headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
            
            response = await self.client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            if not content or content.strip() == '':
                raise ValueError("Empty response from API")
            
            return content
            
        except httpx.TimeoutException as e:
            logger.error(f"AIPipe API timeout: {e}")
            raise Exception("API request timed out")
        except httpx.HTTPError as e:
            logger.error(f"AIPipe HTTP error: {e}")
            raise Exception(f"HTTP error: {str(e)}")
        except Exception as e:
            logger.error(f"AIPipe API error: {e}")
            raise Exception(f"API error: {str(e)}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

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
                # Try to read as a table first
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                    # If it has more than one column, treat it as tabular
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
        """Extract text from PDF"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
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
        
    def __del__(self):
        """Cleanup temp directory"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")
        
    async def analyze(self, questions: str, files: List[UploadFile]) -> Any:
        """Main analysis pipeline"""
        try:
            # Step 1: Process all files
            processed_files = await self._process_files(files)
            
            # Step 2: Extract URLs from questions and scrape if needed
            urls = re.findall(r'https?://[^\s\)]+', questions)
            scraped_data = []
            for url in urls:
                scraped = await WebScraper.scrape_url(url)
                scraped_data.append(scraped)
            
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
        
        for file in files:
            if file.filename == "questions.txt":
                continue
                
            # Save file temporarily
            file_path = os.path.join(self.temp_dir, file.filename)
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Process file
            file_info = await FileProcessor.process_file(file_path, file.filename)
            processed.append(file_info)
            
        return processed
    
    async def _create_execution_plan(self, context: Dict) -> Dict:
        """Create a flexible execution plan based on context"""
        
        # Determine if the context involves CSV data to adjust the prompt
        is_csv_analysis = any(f.get('type') == 'tabular' for f in context.get('files', []))
        
        # Generic prompt for initial planning
        plan_prompt = f"""You are an expert data analyst AI. Analyze the user's questions and available data to create a comprehensive JSON execution plan.

**User's Questions:**
{context['questions']}

**Available Data:**
- Files: {[f.get('filename', 'unknown') for f in context['files']]}
- File Types: {[f.get('type', 'unknown') for f in context['files']]}
- Scraped URLs: {[s.get('url', 'unknown') for s in context['scraped_data']]}

**Detailed Context:**
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

**Instructions:**
Create a JSON execution plan with a list of steps.
- Each step must have a unique `step_id`, an `action`, a `description`, and `params`.
- Actions can be: `run_sql`, `statistical_analysis`, `create_visualization`.
- For `run_sql`, provide a valid SQL query to run against the loaded data, which will be available in a table called `main_data`.
- For `create_visualization`, specify `plot_type` and `columns`.

**Example for a CSV file asking for total revenue:**
```json
{{
  "steps": [
    {{
      "step_id": "step_1_total_revenue",
      "action": "run_sql",
      "description": "Calculate the total revenue by summing the product of Price and Quantity.",
      "params": {{
        "query": "SELECT SUM(Price * Quantity) AS total_revenue FROM main_data;"
      }},
      "expected_output": "A single value representing the total revenue."
    }}
  ]
}}
Now, based on the user's questions and the provided data, generate the precise JSON execution plan. Respond with valid JSON only.
"""

    try:
        plan_text = await self.llm.generate_response(plan_prompt, json_mode=True)
        # Clean up response if needed
        plan_text = re.sub(r'^```json\s*', '', plan_text)
        plan_text = re.sub(r'\s*```$', '', plan_text)
        
        plan = json.loads(plan_text)
        logger.info(f"Generated execution plan: {plan}")
        return plan
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse plan JSON: {e}")
        logger.error(f"Raw response: {plan_text}")
        raise ValueError("Failed to generate a valid execution plan.")
    except Exception as e:
        logger.error(f"Error creating execution plan: {e}")
        raise

async def _execute_plan(self, plan: Dict, context: Dict) -> Dict:
    """Execute the analysis plan"""
    results = {}
    
    # Initialize DuckDB connection
    con = duckdb.connect(database=':memory:', read_only=False)
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")

    try:
        main_df = None
        
        # First, handle local files if any exist
        for file_info in context['files']:
            if file_info.get('type') == 'tabular' and file_info.get('data') is not None:
                try:
                    df = file_info['data']
                    df = self._clean_dataframe(df)
                    table_name = f"file_{re.sub(r'[^A-Za-z0-9_]', '', Path(file_info['filename']).stem)}"
                    con.register(table_name, df)
                    # Use the first tabular file as the main dataframe
                    if main_df is None:
                        main_df = df
                        con.register('main_data', df)
                        logger.info(f"Registered file '{file_info['filename']}' as main_data with shape: {df.shape}")
                except Exception as e:
                    logger.error(f"Error processing file {file_info['filename']}: {e}")

        # Handle scraped data
        for i, scraped in enumerate(context.get('scraped_data', [])):
            if 'tables' in scraped and scraped['tables']:
                for table_info in scraped['tables']:
                    try:
                        df = pd.DataFrame(table_info['data'])
                        df = self._clean_dataframe(df)
                        table_name = f"scraped_table_{i}_{table_info['index']}"
                        con.register(table_name, df)
                        if main_df is None:
                            main_df = df
                            con.register('main_data', df)
                            logger.info(f"Registered scraped table as main_data with shape: {df.shape}")
                    except Exception as e:
                        logger.error(f"Error processing scraped table: {e}")
        
        if main_df is None and 's3://' not in str(plan):
             raise ValueError("No tabular data (CSV, Excel, etc.) was found to analyze.")

        # Execute plan steps
        for step in plan.get('steps', []):
            step_id = step['step_id']
            action = step['action']
            params = step.get('params', {})
            
            logger.info(f"Executing {step_id}: {action}")
            
            try:
                if action == 'run_sql':
                    query = params.get('query', '')
                    if not query:
                        raise ValueError("SQL query is empty.")
                    
                    logger.info(f"Executing SQL: {query}")
                    result_df = con.execute(query).df()
                    results[step_id] = {
                        'type': 'dataframe',
                        'data': result_df.to_dict('records'),
                        'columns': list(result_df.columns),
                        'shape': result_df.shape
                    }

                elif action == 'statistical_analysis':
                    if main_df is None: 
                        raise ValueError("Statistical analysis requires a loaded dataframe.")
                    
                    analysis_df = main_df.copy()
                    analysis_result = await self._perform_statistical_analysis(analysis_df, params)
                    results[step_id] = analysis_result
                
                elif action == 'create_visualization':
                    data_source_step = params.get("data_source_step")
                    
                    # Use data from a previous SQL step if specified
                    if data_source_step and data_source_step in results:
                        plot_data = results[data_source_step]
                        if plot_data.get("type") == "dataframe":
                            plot_df = pd.DataFrame(plot_data['data'])
                        else:
                            raise ValueError(f"Source step {data_source_step} did not produce a dataframe.")
                    elif main_df is not None:
                        plot_df = main_df.copy()
                    else:
                        raise ValueError("No data available for visualization.")
                    
                    plot_result = await self._create_visualization(plot_df, params)
                    results[step_id] = plot_result
                
            except Exception as e:
                logger.error(f"Error in step {step_id}: {e}")
                results[step_id] = {'type': 'error', 'error': str(e)}
        
    finally:
        con.close()
        
    return results

def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize dataframe columns for SQL compatibility."""
    # Clean column names: remove special characters, replace spaces with underscores
    clean_cols = {}
    for col in df.columns:
        new_col_name = re.sub(r'[^A-Za-z0-9_]+', '', col)
        new_col_name = re.sub(r'\s+', '_', new_col_name).strip('_')
        clean_cols[col] = new_col_name
    df = df.rename(columns=clean_cols)
    
    # Attempt to convert object columns to numeric where appropriate
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try converting to numeric, coercing errors
            df[col] = pd.to_numeric(df[col], errors='ignore')
    
    return df

def _clean_sql_query(self, query: str, available_columns: List[str]) -> str:
    """Clean and validate SQL query"""
    # Replace common column name patterns with actual column names
    for col in available_columns:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(col), re.IGNORECASE)
        query = pattern.sub(f'"{col}"', query)
    
    # Ensure we're querying the main table
    if 'FROM' not in query.upper() and 'from' not in query.lower():
        query = query + " FROM main_data"
    
    return query

async def _perform_statistical_analysis(self, df: pd.DataFrame, params: Dict) -> Dict:
    """Perform statistical analysis"""
    try:
        analysis_type = params.get('analysis_type', 'correlation')
        columns = params.get('columns', [])
        
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty:
            return {'type': 'error', 'error': 'No numeric columns available for statistical analysis.'}

        if analysis_type == 'regression' and len(columns) >= 2:
            x_col, y_col = columns[0], columns[1]
            if x_col not in numeric_df.columns or y_col not in numeric_df.columns:
                raise ValueError(f"Columns {x_col}, {y_col} not found or not numeric.")
            
            clean_data = numeric_df[[x_col, y_col]].dropna()
            if len(clean_data) < 2:
                return {'type': 'error', 'error': 'Not enough data for regression analysis.'}

            slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data[x_col], clean_data[y_col])
            return {
                'type': 'regression',
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value**2)
            }
        
        elif analysis_type == 'correlation':
            if len(numeric_df.columns) < 2:
                return {'type': 'error', 'error': 'Not enough numeric columns for correlation matrix.'}
            corr_matrix = numeric_df.corr()
            return {'type': 'correlation', 'correlation_matrix': corr_matrix.to_dict()}
        
        return {'type': 'error', 'error': f'Unsupported analysis type: {analysis_type}'}
        
    except Exception as e:
        logger.error(f"Statistical analysis error: {e}")
        return {'type': 'statistical_analysis', 'error': str(e)}

async def _create_visualization(self, df: pd.DataFrame, params: Dict) -> Dict:
    """Create visualizations"""
    try:
        plot_type = params.get('plot_type', 'bar')
        columns = params.get('columns', [])
        
        if len(columns) < 1:
            return {'type': 'visualization', 'error': 'Columns for plotting were not specified.'}

        x_col = columns[0]
        y_col = columns[1] if len(columns) > 1 else None

        if x_col not in df.columns:
            return {'type': 'visualization', 'error': f"X-axis column '{x_col}' not found in data."}
        if y_col and y_col not in df.columns:
            return {'type': 'visualization', 'error': f"Y-axis column '{y_col}' not found in data."}
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'bar':
            if not y_col: # If only one column, do a value count
                plot_data = df[x_col].value_counts().sort_index()
                sns.barplot(x=plot_data.index, y=plot_data.values)
                plt.ylabel("Count")
            else: # If two columns, treat as x and y
                # If x is numeric, it might need to be grouped
                if pd.api.types.is_numeric_dtype(df[x_col]) and df[x_col].nunique() > 20:
                     return {'type': 'visualization', 'error': f"Column '{x_col}' has too many unique numeric values for a bar chart."}
                
                # Sort by the y-value for better presentation
                plot_data = df.sort_values(by=y_col, ascending=False)
                sns.barplot(x=x_col, y=y_col, data=plot_data)

        elif plot_type == 'scatter':
            if not y_col:
                return {'type': 'visualization', 'error': 'Scatter plot requires two columns (x and y).'}
            sns.scatterplot(x=x_col, y=y_col, data=df)
            if params.get('add_regression', False):
                sns.regplot(x=x_col, y=y_col, data=df, scatter=False, color='red', line_kws={'linestyle':'--'})
        
        elif plot_type == 'line':
            if not y_col:
                return {'type': 'visualization', 'error': 'Line plot requires two columns (x and y).'}
            sns.lineplot(x=x_col, y=y_col, data=df, marker='o')

        else:
            return {'type': 'visualization', 'error': f"Unsupported plot type: {plot_type}"}

        plt.xlabel(x_col.replace('_', ' ').title(), fontsize=12)
        if y_col:
            plt.ylabel(y_col.replace('_', ' ').title(), fontsize=12)
        
        title_y = f" vs. {y_col.replace('_', ' ').title()}" if y_col else ""
        plt.title(f"{plot_type.title()} of {x_col.replace('_', ' ').title()}{title_y}", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close()
        img_data = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            'type': 'visualization',
            'format': 'base64_png',
            'data': f"data:image/png;base64,{img_data}",
            'plot_info': { 'type': plot_type, 'x_column': x_col, 'y_column': y_col }
        }
        
    except Exception as e:
        logger.error(f"Visualization error: {e}\n{traceback.format_exc()}")
        return {'type': 'visualization', 'error': str(e)}

async def _generate_final_answer(self, questions: str, results: Dict, context: Dict) -> Any:
    """Generate the final answer by synthesizing analysis results."""
    
    synthesis_prompt = f"""You are a data synthesis AI. Your task is to provide the final answer in the EXACT JSON format requested by the user, based on the provided analysis results.
Original User Questions:
{questions}

Available Analysis Results:

JSON

{json.dumps(results, indent=2, default=str)}
CRITICAL INSTRUCTIONS:

Carefully examine the original questions to understand the required JSON keys (e.g., "total_revenue", "top_selling_product", "sales_by_region_chart").

Extract the specific answers from the "Available Analysis Results" section above. Find the data within the data field of the corresponding step.

For numerical answers (like total revenue), provide the number directly, not a list or object.

For text answers (like the top product), provide the string value.

For visualization requests, find the corresponding 'visualization' result and include the complete base64 data URI string.

Construct a single JSON object using the exact keys requested by the user. Do not add extra commentary, explanations, or markdown.

Generate a valid JSON object as the final answer.
"""

    try:
        # Generate the response from the LLM
        final_response_text = await self.llm.generate_response(synthesis_prompt, json_mode=True)
        
        # Clean up potential markdown formatting from the LLM's response
        cleaned_response = re.sub(r'^```json\s*', '', final_response_text, flags=re.MULTILINE)
        cleaned_response = re.sub(r'\s*```$', '', cleaned_response, flags=re.MULTILINE)

        # Try to parse the cleaned response as JSON
        return json.loads(cleaned_response)

    except json.JSONDecodeError:
        logger.error(f"Failed to parse the final JSON response from the LLM. Raw response: {final_response_text}")
        # If parsing fails, return an error object with the raw text to help with debugging
        raise HTTPException(
            status_code=500, 
            detail={"error": "The AI failed to generate a valid JSON response.", "raw_response": final_response_text}
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during final answer synthesis: {e}\n{traceback.format_exc()}")
        # Re-raise other exceptions to be caught by the main FastAPI error handler
        raise
Initialize FastAPI app
app = FastAPI(
title="Universal Data Analyst Agent",
description="AI-powered data analysis with support for multiple file formats, web scraping, and visualizations",
version="1.0.0"
)

Add CORS middleware
app.add_middleware(
CORSMiddleware,
allow_origins=[""],
allow_credentials=True,
allow_methods=[""],
allow_headers=["*"],
)

Validate configuration on startup
Config.validate()

Initialize LLM provider
def get_llm_provider():
if Config.DEFAULT_LLM_PROVIDER == 'gemini' and Config.GEMINI_API_KEY:
return GeminiProvider(Config.GEMINI_API_KEY)
elif Config.DEFAULT_LLM_PROVIDER == 'aipipe' and Config.AIPIPE_TOKEN:
return AIPipeProvider(Config.AIPIPE_TOKEN, model=Config.AIPIPE_MODEL)
elif Config.DEFAULT_LLM_PROVIDER == 'ollama':
return OllamaProvider(Config.OLLAMA_BASE_URL, Config.OLLAMA_MODEL)
else:
# Fallback to Ollama if no other provider is available
logger.warning("No primary LLM provider configured, falling back to Ollama.")
return OllamaProvider(Config.OLLAMA_BASE_URL, Config.OLLAMA_MODEL)

API Routes
@app.get("/")
async def root():
"""Root endpoint that serves the main HTML interface"""
if Path("index.html").exists():
return FileResponse("index.html")
return HTMLResponse("<html><body><h1>Data Analyst Agent is running</h1><p>Upload files to /api/ endpoint.</p></body></html>")

@app.get("/health")
async def health_check():
"""Health check endpoint"""
try:
# Test LLM provider initialization
get_llm_provider()
return {
"status": "healthy",
"timestamp": datetime.now().isoformat(),
"llm_provider": Config.DEFAULT_LLM_PROVIDER,
"version": "1.0.0"
}
except Exception as e:
logger.error(f"Health check failed: {e}")
return JSONResponse(
status_code=503,
content={
"status": "unhealthy",
"error": str(e),
"timestamp": datetime.now().isoformat()
}
)

@app.get("/config")
async def get_config():
"""Get system configuration"""
return {
"llm_provider": Config.DEFAULT_LLM_PROVIDER,
"ai_pipe_model": Config.AIPIPE_MODEL,
"max_file_size_mb": Config.MAX_FILE_SIZE // (1024 * 1024),
"request_timeout_seconds": Config.REQUEST_TIMEOUT,
"has_gemini_key": bool(Config.GEMINI_API_KEY),
"has_aipipe_token": bool(Config.AIPIPE_TOKEN),
"ollama_model": Config.OLLAMA_MODEL,
"ollama_base_url": Config.OLLAMA_BASE_URL
}

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
"""Main analysis endpoint"""
start_time = datetime.now()

try:
    # Extract questions from files
    questions_content = None
    data_files = []
    
    for file in files:
        if file.filename == "questions.txt":
            questions_content = (await file.read()).decode('utf-8')
        else:
            # Reset file pointer for other files
            await file.seek(0)
            data_files.append(file)
    
    if not questions_content:
        raise HTTPException(status_code=400, detail="A 'questions.txt' file is required.")
    
    # Validate file sizes
    for file in data_files:
        if file.size and file.size > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File {file.filename} exceeds maximum size of {Config.MAX_FILE_SIZE // (1024*1024)}MB"
            )
    
    # Initialize agent and analyze
    llm_provider = get_llm_provider()
    agent = DataAnalystAgent(llm_provider)
    
    logger.info(f"Starting analysis with {len(data_files)} files...")
    logger.info(f"Questions: {questions_content[:300]}...")
    
    result = await agent.analyze(questions_content, data_files)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Analysis completed in {processing_time:.2f} seconds")
    
    return result
    
except HTTPException:
    # Re-raise HTTP exceptions directly
    raise
except Exception as e:
    logger.error(f"A critical error occurred in the analysis endpoint: {e}\n{traceback.format_exc()}")
    raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
Optional: Add a simple file upload endpoint for testing
@app.post("/upload-test")
async def upload_test(files: List[UploadFile] = File(...)):
"""Test endpoint to check file uploads"""
file_info = []
for file in files:
content = await file.read()
file_info.append({
"filename": file.filename,
"content_type": file.content_type,
"size": len(content)
})
return {"uploaded_files": file_info}

Error handlers
@app.exception_handler(413)
async def file_too_large_handler(request, exc):
return JSONResponse(
status_code=413,
content={"error": "File too large", "detail": f"Maximum size is {Config.MAX_FILE_SIZE // (1024*1024)}MB."}
)

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
logger.error(f"Caught an internal server error: {exc}")
# The detail should come from the raised HTTPException
detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
return JSONResponse(
status_code=500,
content={"error": "Internal Server Error", "detail": detail}
)

Serve static files (if an index.html exists)
if Path("index.html").exists():
app.mount("/", StaticFiles(directory=".", html=True), name="static")

Development server runner
if name == "main":
import sys

# Check for required dependencies
try:
    import uvicorn
except ImportError:
    print("uvicorn not found. Install with: pip install uvicorn")
    sys.exit(1)

# Set up logging for development
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("\n🚀 Starting Universal Data Analyst Agent...")
print(f"📊 LLM Provider: {Config.DEFAULT_LLM_PROVIDER}")
print(f"🤖 AI Model: {Config.AIPIPE_MODEL if Config.DEFAULT_LLM_PROVIDER == 'aipipe' else 'N/A'}")
print(f"📁 Max File Size: {Config.MAX_FILE_SIZE // (1024*1024)}MB")
print(f"⏱️  Request Timeout: {Config.REQUEST_TIMEOUT}s")
print("\n🌐 Server will be available at: http://localhost:8000")
print("📖 API Documentation: http://localhost:8000/docs")
print("🔍 Health Check: http://localhost:8000/health")

uvicorn.run(
    "app:app",
    host="0.0.0.0",
    port=8000,
    reload=True,
    access_log=True
)
