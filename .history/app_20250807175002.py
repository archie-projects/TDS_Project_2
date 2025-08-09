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
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')
    DEFAULT_LLM_PROVIDER = os.getenv('DEFAULT_LLM_PROVIDER', 'gemini')  # gemini, aipipe, ollama
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
        self.client = httpx.AsyncClient(timeout=180.0)
        
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
            urls = re.findall(r'https?://[^\s]+', questions)
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

Create a JSON execution plan. Analyze the questions to determine:
1. What type of analysis is needed (counting, correlation, regression, comparison, etc.)
2. What response format is expected (JSON array, JSON object, specific structure)
3. What visualizations are needed (scatter plots, bar charts, etc.)

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
- Include SQL steps for data manipulation, filtering, and aggregation
- Include statistical analysis steps for correlations, regressions, etc.
- Include visualization steps for any requested plots
- Plan for the exact response format the user expects
- If questions mention specific numbers (like $2bn, $1.5bn), plan filtering steps
- For time-series data, consider date filtering
- For plotting, specify exact columns and styling requirements

Respond with valid JSON only."""
        
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
            # Fallback simple plan
            return {
                "analysis_type": "descriptive",
                "expected_response_format": "json_array",
                "steps": [
                    {"step_id": "step_1", "action": "load_data", "description": "Load available data", "params": {}}
                ]
            }
    
    async def _execute_plan(self, plan: Dict, context: Dict) -> Dict:
        """Execute the analysis plan"""
        results = {}
        
        # Initialize DuckDB connection
        con = duckdb.connect(database=':memory:', read_only=False)
        con.execute("INSTALL httpfs; LOAD httpfs;")
        
        try:
            main_df = None
            
            # Load scraped data first
            for i, scraped in enumerate(context['scraped_data']):
                if 'tables' in scraped and scraped['tables']:
                    for j, table_info in enumerate(scraped['tables']):
                        try:
                            df = pd.DataFrame(table_info['data'])
                            df = self._clean_dataframe(df)
                            table_name = f"scraped_table_{i}_{j}"
                            con.register(table_name, df)
                            if main_df is None:
                                main_df = df
                                con.register('main_data', df)
                                logger.info(f"Registered scraped table as main_data with shape: {df.shape}")
                        except Exception as e:
                            logger.error(f"Error processing scraped table: {e}")
            
            # Load file data
            for file_info in context['files']:
                if file_info.get('type') == 'tabular' and file_info.get('data') is not None:
                    try:
                        df = file_info['data']
                        df = self._clean_dataframe(df)
                        clean_filename = re.sub(r'[^\w]', '_', file_info['filename'])
                        table_name = f"file_{clean_filename}"
                        con.register(table_name, df)
                        if main_df is None:
                            main_df = df
                            con.register('main_data', df)
                            logger.info(f"Registered file as main_data with shape: {df.shape}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_info['filename']}: {e}")
            
            if main_df is None:
                logger.warning("No data available for analysis")
                return {"error": "No suitable data found for analysis"}
            
            # Execute plan steps
            for step in plan.get('steps', []):
                step_id = step['step_id']
                action = step['action']
                params = step.get('params', {})
                
                logger.info(f"Executing {step_id}: {action}")
                
                try:
                    if action == 'run_sql':
                        query = params.get('query', '')
                        if query:
                            # Clean and validate SQL
                            query = self._clean_sql_query(query, list(main_df.columns))
                            logger.info(f"Executing SQL: {query}")
                            result_df = con.execute(query).df()
                            results[step_id] = {
                                'type': 'dataframe',
                                'data': result_df.to_dict('records'),
                                'columns': list(result_df.columns),
                                'shape': result_df.shape
                            }
                    
                    elif action == 'statistical_analysis':
                        analysis_result = await self._perform_statistical_analysis(main_df, params)
                        results[step_id] = analysis_result
                    
                    elif action == 'create_visualization':
                        # Use result from previous step if specified
                        plot_data = main_df
                        if 'data_source_step' in params and params['data_source_step'] in results:
                            step_data = results[params['data_source_step']]
                            if step_data.get('type') == 'dataframe':
                                plot_data = pd.DataFrame(step_data['data'])
                        
                        plot_result = await self._create_visualization(plot_data, params)
                        results[step_id] = plot_result
                    
                    elif action == 'load_data':
                        results[step_id] = {
                            'type': 'data_summary',
                            'shape': main_df.shape,
                            'columns': list(main_df.columns),
                            'dtypes': {col: str(dtype) for col, dtype in main_df.dtypes.items()},
                            'sample': main_df.head(5).to_dict('records')
                        }
                    
                except Exception as e:
                    logger.error(f"Error in step {step_id}: {e}")
                    results[step_id] = {'type': 'error', 'error': str(e)}
            
        finally:
            con.close()
            
        return results
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataframe"""
        # Clean column names
        df.columns = [re.sub(r'[^\w\s]', '', str(col)).strip().replace(' ', '_') for col in df.columns]
        
        # Convert numeric columns, handling currency and percentages
        for col in df.columns:
            if df[col].dtype == 'object':
                # Handle currency, percentages, and numbers with commas
                cleaned = df[col].astype(str).str.replace(r'[$,%]', '', regex=True)
                numeric_series = pd.to_numeric(cleaned, errors='coerce')
                if not numeric_series.isna().all():
                    df[col] = numeric_series
        
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
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if analysis_type == 'correlation' and len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                
                # Find strongest correlation (excluding diagonal)
                corr_values = corr_matrix.abs().unstack()
                corr_values = corr_values[corr_values < 1.0]  # Remove diagonal
                if not corr_values.empty:
                    strongest_idx = corr_values.idxmax()
                    strongest_value = corr_values.max()
                else:
                    strongest_idx = ('N/A', 'N/A')
                    strongest_value = 0
                
                return {
                    'type': 'correlation',
                    'correlation_matrix': corr_matrix.to_dict(),
                    'strongest_correlation': {
                        'columns': strongest_idx,
                        'value': float(strongest_value)
                    },
                    'numeric_columns': numeric_cols
                }
            
            elif analysis_type == 'regression' and len(numeric_cols) >= 2:
                x_col = params.get('x_column', numeric_cols[0])
                y_col = params.get('y_column', numeric_cols[1])
                
                if x_col in df.columns and y_col in df.columns:
                    # Remove NaN values
                    clean_data = df[[x_col, y_col]].dropna()
                    if len(clean_data) > 1:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            clean_data[x_col], clean_data[y_col]
                        )
                        return {
                            'type': 'regression',
                            'x_column': x_col,
                            'y_column': y_col,
                            'slope': float(slope),
                            'intercept': float(intercept),
                            'r_squared': float(r_value**2),
                            'p_value': float(p_value),
                            'sample_size': len(clean_data)
                        }
            
            return {
                'type': 'statistical_analysis', 
                'error': f'Cannot perform {analysis_type} analysis',
                'available_numeric_columns': numeric_cols
            }
            
        except Exception as e:
            logger.error(f"Statistical analysis error: {e}")
            return {'type': 'statistical_analysis', 'error': str(e)}
    
    async def _create_visualization(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Create visualizations"""
        try:
            plot_type = params.get('plot_type', 'scatter')
            
            # Auto-detect columns if not specified
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            x_col = params.get('x_column') or params.get('columns', [None])[0]
            y_col = params.get('y_column') or (params.get('columns', [None, None])[1] if len(params.get('columns', [])) > 1 else None)
            
            # Fallback to first two numeric columns
            if not x_col and len(numeric_cols) > 0:
                x_col = numeric_cols[0]
            if not y_col and len(numeric_cols) > 1:
                y_col = numeric_cols[1]
            
            if not x_col or x_col not in df.columns:
                return {'type': 'visualization', 'error': f'X column {x_col} not found'}
            if plot_type == 'scatter' and (not y_col or y_col not in df.columns):
                return {'type': 'visualization', 'error': f'Y column {y_col} not found'}
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.style.use('default')
            
            if plot_type == 'scatter':
                plt.scatter(df[x_col], df[y_col], alpha=0.6, s=50)
                
                # Add regression line if requested
                if params.get('add_regression', True):
                    # Clean data for regression
                    clean_data = df[[x_col, y_col]].dropna()
                    if len(clean_data) > 1:
                        z = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(clean_data[x_col].min(), clean_data[x_col].max(), 100)
                        plt.plot(x_line, p(x_line), 
                                color=params.get('line_color', 'red'), 
                                linestyle=params.get('line_style', '--'),
                                linewidth=2, label='Regression Line')
                        plt.legend()
            
            elif plot_type == 'bar':
                if df[x_col].dtype == 'object' or df[x_col].nunique() < 20:
                    # Categorical bar chart
                    if y_col and y_col in df.columns:
                        df_grouped = df.groupby(x_col)[y_col].sum().sort_values(ascending=False)
                    else:
                        df_grouped = df[x_col].value_counts()
                    
                    plt.bar(range(len(df_grouped)), df_grouped.values)
                    plt.xticks(range(len(df_grouped)), df_grouped.index, rotation=45, ha='right')
                else:
                    plt.bar(df[x_col], df[y_col] if y_col else [1]*len(df))
            
            elif plot_type == 'line':
                y_data = df[y_col] if y_col else df[x_col]
                plt.plot(df[x_col], y_data, marker='o', linewidth=2, markersize=4)
            
            # Styling
            plt.xlabel(x_col, fontsize=12)
            if y_col and plot_type in ['scatter', 'line']:
                plt.ylabel(y_col, fontsize=12)
            plt.title(f"{plot_type.title()} Plot: {x_col}" + (f" vs {y_col}" if y_col and plot_type == 'scatter' else ""), fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            img_data = base64.b64encode(buffer.getvalue()).decode()
            
            # Check size limit (100KB)
            if len(img_data) > 100000:
                logger.warning(f"Image size {len(img_data)} exceeds limit")
                return {'type': 'visualization', 'error': 'Generated image exceeds size limit'}
            
            return {
                'type': 'visualization',
                'format': 'base64_png',
                'data': f"data:image/png;base64,{img_data}",
                'size_bytes': len(img_data),
                'plot_info': {
                    'type': plot_type,
                    'x_column': x_col,
                    'y_column': y_col,
                    'data_points': len(df)
                }
            }
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return {'type': 'visualization', 'error': str(e)}
    
    async def _generate_final_answer(self, questions: str, results: Dict, context: Dict) -> Any:
        """Generate the final answer in the requested format"""
        
        synthesis_prompt = f"""You are a data synthesis AI. Your task is to provide the EXACT final answer format requested by the user.

ORIGINAL QUESTIONS:
{questions}

ANALYSIS RESULTS:
{json.dumps(results, indent=2, default=str)}

CRITICAL INSTRUCTIONS:
1. Examine the questions to determine the EXACT response format expected
2. If questions ask for a JSON array like [value1, value2, value3], respond with that EXACT format
3. If questions ask for a JSON object like {{"question1": "answer1"}}, respond with that EXACT format
4. Extract specific values from the analysis results to answer each question
5. For visualizations, include the complete base64 data URI from the results
6. For numerical answers, use the exact numbers from the analysis (don't round unless specified)
7. For correlation questions, extract the correlation coefficient value
8. For regression questions, extract the slope value
9. If analysis failed for any question, use null or appropriate fallback

RESPONSE FORMAT EXAMPLES:
- JSON Array: [1, "Titanic", 0.485782, "data:image/png;base64,..."]
- JSON Object: {{"Which court disposed most cases?": "High Court XYZ", "Regression slope": 0.123}}
- Single value: 42 (if only one answer requested)

IMPORTANT: 
- Respond with ONLY the final JSON - no explanations, no markdown formatting
- Ensure all string values are properly quoted
- Ensure numbers are not quoted unless they should be strings
- Include complete data URIs for any requested visualizations
- Match the exact structure and format the user expects

Generate the final answer now:
"""

        try:
            final_response = await self.llm.generate_response(synthesis_prompt, json_mode=True)
            # Clean up response if needed
            final_response = re.sub(r'^```json\s*', '', final_response)
            final_response = re.sub(r'\s*```, '', final_response)
            
            # Try to parse as JSON to validate
            try:
                parsed_response = json.loads(final_response)
                return parsed_response
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw response
                return final_response
                
        except Exception as e:
            logger.error(f"Final answer generation failed: {e}")
            return {"error": f"Could not generate final answer: {str(e)}"}

# Initialize FastAPI app
app = FastAPI(
    title="Universal Data Analyst Agent",
    description="AI-powered data analysis with support for multiple file formats, web scraping, and visualizations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate configuration on startup
Config.validate()

# Initialize LLM provider
def get_llm_provider():
    if Config.DEFAULT_LLM_PROVIDER == 'gemini' and Config.GEMINI_API_KEY:
        return GeminiProvider(Config.GEMINI_API_KEY)
    elif Config.DEFAULT_LLM_PROVIDER == 'aipipe' and Config.AIPIPE_TOKEN:
        return AIPipeProvider(Config.AIPIPE_TOKEN)
    elif Config.DEFAULT_LLM_PROVIDER == 'ollama':
        return OllamaProvider(Config.OLLAMA_BASE_URL, Config.OLLAMA_MODEL)
    else:
        # Fallback to Ollama if no other provider is available
        return OllamaProvider(Config.OLLAMA_BASE_URL, Config.OLLAMA_MODEL)

# API Routes
@app.get("/")
async def root():
    """Root endpoint that serves the main HTML interface"""
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test LLM provider
        llm = get_llm_provider()
        # Simple test - don't actually call the LLM to avoid costs
        
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
            raise HTTPException(status_code=400, detail="No questions.txt file provided")
        
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
        
        logger.info(f"Starting analysis with {len(data_files)} files")
        logger.info(f"Questions: {questions_content[:200]}...")
        
        result = await agent.analyze(questions_content, data_files)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Optional: Add a simple file upload endpoint for testing
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

# Error handlers
@app.exception_handler(413)
async def file_too_large_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={"error": "File too large", "max_size_mb": Config.MAX_FILE_SIZE // (1024*1024)}
    )

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Serve static files (including the HTML interface)
if Path("index.html").exists():
    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())

# Development server runner
if __name__ == "__main__":
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
            )
import os
import re
import json
