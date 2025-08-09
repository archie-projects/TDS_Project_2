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
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
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
import mimetypes

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
            logger.warning("GEMINI_API_KEY environment variable not set. Some features may not work.")
            return False
        return True

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
        # Use gemini-1.5-flash as requested
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    async def generate_response(self, prompt: str, json_mode: bool = False) -> str:
        try:
            if json_mode:
                prompt += "\n\nIMPORTANT: Respond with valid JSON only, no other text or markdown formatting. Do not include ```json or ``` markers."

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error during plan execution: {e}")
            results['execution_error'] = {'type': 'error', 'error': str(e)}
        finally:
            con.close()

        return results

    async def _generate_final_answer(self, questions: str, results: Dict, context: Dict, plan: Dict) -> Any:
        """Generate the final answer in the requested format"""

        # Handle simple knowledge-based responses
        if plan.get("analysis_type") == "knowledge_based":
            for step_id, result in results.items():
                if result.get('type') == 'knowledge_response':
                    return {"answer": result.get('answer', 'No answer found.')}

        # For data analysis, synthesize all results
        synthesis_prompt = f"""You are a data analyst. Provide the final answer based on the execution results.

ORIGINAL QUESTIONS:
{questions}

EXECUTION RESULTS:
{json.dumps(results, indent=2, default=str)}

INSTRUCTIONS:
1. Answer the user's questions directly based on the execution results
2. If the user requested a specific format (JSON array, JSON object), follow it exactly
3. Extract key values from dataframe results
4. Include visualization data URIs if created
5. Return valid JSON only, no markdown

Generate the final answer now."""

        try:
            final_response_str = await self.llm.generate_response(synthesis_prompt, json_mode=True)
            final_response_str = re.sub(r'^```json\s*', '', final_response_str)
            final_response_str = re.sub(r'\s*```f"Gemini API error: {e}")
            raise

class WebScraper:
    """Handles web scraping with various strategies"""

    @staticmethod
    async def scrape_url(url: str) -> Dict[str, Any]:
        """Scrape data from URL"""
        try:
            # Clean URL first - remove any trailing characters
            url = url.rstrip(')').rstrip('.')

            async with httpx.AsyncClient(timeout=60.0) as client:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = await client.get(url, follow_redirects=True, headers=headers)
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

            # Get file extension and MIME type
            file_ext = Path(filename).suffix.lower()
            mime_type, _ = mimetypes.guess_type(filename)

            # Handle various file types
            if file_ext in ['.csv', '.tsv'] or (mime_type and 'csv' in mime_type):
                file_info["type"] = "tabular"
                # Try different encodings and separators
                encodings = ['utf-8', 'latin-1', 'cp1252']
                separators = [',', '\t', ';', '|']
                
                for encoding in encodings:
                    for sep in separators:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding, sep=sep, on_bad_lines='skip')
                            if df.shape[1] > 1:  # Valid table
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

            elif file_ext in ['.xlsx', '.xls'] or (mime_type and 'spreadsheet' in mime_type):
                file_info["type"] = "tabular"
                try:
                    df = pd.read_excel(file_path)
                    file_info["data"] = df
                    file_info["metadata"] = {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                    }
                except Exception as e:
                    logger.error(f"Excel processing error: {e}")

            elif file_ext == '.json' or (mime_type and 'json' in mime_type):
                file_info["type"] = "json"
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    file_info["data"] = data
                    file_info["metadata"] = {"keys": list(data.keys()) if isinstance(data, dict) else "array"}
                except Exception:
                    # Try with different encodings
                    for encoding in ['latin-1', 'cp1252']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                data = json.load(f)
                            file_info["data"] = data
                            file_info["metadata"] = {"keys": list(data.keys()) if isinstance(data, dict) else "array"}
                            break
                        except Exception:
                            continue

            elif file_ext == '.pdf' or (mime_type and 'pdf' in mime_type):
                file_info["type"] = "text"
                text = await FileProcessor._extract_pdf_text(file_path)
                file_info["data"] = text
                file_info["metadata"] = {"length": len(text), "pages": text.count('\n\n') + 1}

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

            elif file_ext in ['.txt', '.md', '.rst'] or (mime_type and 'text' in mime_type):
                # Try to read as a table first, then as plain text
                encodings = ['utf-8', 'latin-1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        # First attempt: structured data
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
                
                # If not tabular, read as plain text
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

            else:
                # Unknown file type - try to process as text
                file_info["type"] = "unknown"
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read(1000)  # Read first 1000 chars
                    if text.isprintable():
                        file_info["type"] = "text"
                        file_info["data"] = text
                        file_info["metadata"] = {"note": "Unknown text file type", "preview_length": len(text)}
                except Exception:
                    with open(file_path, 'rb') as f:
                        data = f.read(100)  # Read first 100 bytes
                    file_info["metadata"] = {"note": "Binary file - not processed", "size_bytes": os.path.getsize(file_path)}

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

    async def analyze(self, questions: str, files: List[UploadFile] = None) -> Any:
        """Main analysis pipeline"""
        try:
            # Step 1: Process all files (if any)
            processed_files = []
            if files:
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
                "questions": questions,
                "has_data": bool(processed_files or scraped_data)
            }

            # Step 4: Generate execution plan
            plan = await self._create_execution_plan(context)

            # Step 5: Execute the plan
            execution_results = await self._execute_plan(plan, context)

            # Step 6: Generate final response
            final_answer = await self._generate_final_answer(questions, execution_results, context, plan)

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

        data_summary = ""
        if context['files']:
            data_summary += f"Files: {len(context['files'])} files\n"
            for f in context['files']:
                data_summary += f"  - {f.get('filename', 'unknown')} ({f.get('type', 'unknown')})\n"
        
        if context['scraped_data']:
            data_summary += f"Scraped URLs: {len(context['scraped_data'])}\n"

        plan_prompt = f"""You are an expert data analyst. Create a comprehensive execution plan to answer the user's questions.

QUESTIONS TO ANSWER:
{context['questions']}

AVAILABLE DATA:
{data_summary}

DATA AVAILABLE: {context['has_data']}

Create a JSON execution plan. Use 'main_data' as the primary table name for SQL queries.

Response format (JSON only, no markdown):
{{
    "analysis_type": "knowledge_based|data_analysis|web_scraping|mixed",
    "expected_response_format": "json_array|json_object|text",
    "steps": [
        {{
            "step_id": "step_1",
            "action": "knowledge_response|load_data|run_sql|statistical_analysis|create_visualization|web_scraping",
            "description": "what this step does",
            "params": {{
                "query": "SQL query or question",
                "plot_type": "scatter|bar|line",
                "x_col": "column_name",
                "y_col": "column_name",
                "analysis_type": "correlation|regression"
            }}
        }}
    ]
}}"""

        try:
            plan_text = await self.llm.generate_response(plan_prompt, json_mode=True)
            # Clean up response
            plan_text = re.sub(r'^```json\s*', '', plan_text)
            plan_text = re.sub(r'\s*```$', '', plan_text)

            plan = json.loads(plan_text)
            logger.info(f"Generated execution plan: {json.dumps(plan, indent=2)}")
            return plan
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            # Fallback plan
            return {
                "analysis_type": "knowledge_based" if not context['has_data'] else "data_analysis",
                "expected_response_format": "json_object",
                "steps": [
                    {
                        "step_id": "step_1", 
                        "action": "knowledge_response" if not context['has_data'] else "load_data", 
                        "description": "Provide answer based on available information", 
                        "params": {"query": context['questions']}
                    }
                ]
            }

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataframe"""
        # Clean column names
        df.columns = [re.sub(r'[^A-Za-z0-9_]', '', str(col).strip().replace(' ', '_').replace('(', '').replace(')', '')) for col in df.columns]

        for col in df.columns:
            # Attempt to convert to datetime for date-like columns
            if df[col].dtype == 'object':
                if any(term in col.lower() for term in ['date', 'time', 'year']):
                    try:
                        converted_dates = pd.to_datetime(df[col], errors='coerce')
                        if converted_dates.notna().sum() / len(df[col]) > 0.5:
                            df[col] = converted_dates
                            continue
                    except Exception:
                        pass

            # Clean and convert numeric columns
            if df[col].dtype == 'object':
                try:
                    cleaned_series = df[col].astype(str).str.replace(r'[$,%\s]', '', regex=True)
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    if numeric_series.notna().sum() / len(df[col]) > 0.7:
                        df[col] = numeric_series
                except Exception:
                    pass

        return df

    async def _perform_statistical_analysis(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Perform statistical analysis"""
        try:
            analysis_type = params.get('analysis_type', 'correlation')
            columns = params.get('columns', [])

            # Convert datetime columns to numeric for analysis
            numeric_df = df.copy()
            for col in numeric_df.columns:
                if pd.api.types.is_datetime64_any_dtype(numeric_df[col]):
                    numeric_df[col] = numeric_df[col].apply(lambda x: x.toordinal() if pd.notna(x) else np.nan)

            if analysis_type == 'correlation':
                if len(columns) >= 2:
                    numeric_cols = [col for col in columns if col in numeric_df.select_dtypes(include=np.number).columns]
                else:
                    numeric_cols = numeric_df.select_dtypes(include=[np.number]).columns.tolist()

                if len(numeric_cols) >= 2:
                    corr_matrix = numeric_df[numeric_cols].corr()
                    corr_values = corr_matrix.abs().unstack()
                    corr_values = corr_values[corr_values < 1.0]
                    
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

            elif analysis_type == 'regression' and len(columns) >= 2:
                x_col, y_col = columns[0], columns[1]

                if x_col in numeric_df.columns and y_col in numeric_df.columns:
                    clean_data = numeric_df[[x_col, y_col]].dropna()
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

            return {'type': 'statistical_analysis', 'error': f'Cannot perform {analysis_type} analysis.'}

        except Exception as e:
            logger.error(f"Statistical analysis error: {e}")
            return {'type': 'statistical_analysis', 'error': str(e)}

    async def _create_visualization(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Create visualizations"""
        try:
            plot_type = params.get('plot_type', 'bar')
            x_col = params.get('x_col') or params.get('x_axis')
            y_col = params.get('y_col') or params.get('y_axis')
            
            if not x_col:
                cat_cols = df.select_dtypes(include=['object', 'category']).columns
                x_col = cat_cols[0] if len(cat_cols) > 0 else df.columns[0]
            
            if not y_col and plot_type != 'bar':
                num_cols = df.select_dtypes(include=np.number).columns
                y_col = num_cols[0] if len(num_cols) > 0 else None

            if not x_col or x_col not in df.columns:
                return {'type': 'visualization', 'error': f'Column "{x_col}" not found. Available: {list(df.columns)}'}

            # Create plot
            plt.figure(figsize=(12, 8))
            plt.style.use('default')

            plot_data = df.copy()
            plot_cols = [c for c in [x_col, y_col] if c is not None]
            plot_data.dropna(subset=plot_cols, inplace=True)
            
            if plot_data.empty:
                return {'type': 'visualization', 'error': 'No data available for plotting after removing missing values.'}

            if plot_type == 'scatter' and y_col:
                sns.scatterplot(x=x_col, y=y_col, data=plot_data, alpha=0.7)
                if params.get('add_regression', False):
                    sns.regplot(x=x_col, y=y_col, data=plot_data, scatter=False, color='red', linestyle='--')
            elif plot_type == 'bar':
                if y_col:
                    plot_values = plot_data.groupby(x_col)[y_col].sum().sort_values(ascending=False)
                    sns.barplot(x=plot_values.index, y=plot_values.values, palette="viridis")
                else:
                    sns.countplot(x=x_col, data=plot_data, palette="viridis")
            elif plot_type == 'line' and y_col:
                sns.lineplot(x=x_col, y=y_col, data=plot_data, marker='o')

            plt.xlabel(str(x_col).replace('_', ' ').title(), fontsize=14)
            plt.ylabel(str(y_col if y_col else 'Count').replace('_', ' ').title(), fontsize=14)
            plt.xticks(rotation=45, ha='right')
            
            title_y = y_col if y_col else 'Count'
            title = f"{plot_type.title()} of {str(title_y).replace('_',' ').title()} by {str(x_col).replace('_',' ').title()}"
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()

            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            img_data = base64.b64encode(buffer.getvalue()).decode()

            return {
                'type': 'visualization',
                'format': 'base64_png',
                'data': f"data:image/png;base64,{img_data}",
                'size_bytes': len(img_data),
                'plot_info': {
                    'type': plot_type,
                    'x_column': x_col,
                    'y_column': y_col,
                    'title': title
                }
            }

        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return {'type': 'visualization', 'error': str(e)}

    async def _execute_plan(self, plan: Dict, context: Dict) -> Dict:
        """Execute the analysis plan"""
        results = {}

        # Initialize DuckDB connection
        con = duckdb.connect(database=':memory:', read_only=False)
        try:
            main_df = None

            # Load data from files
            for file_info in context['files']:
                if file_info.get('type') == 'tabular' and file_info.get('data') is not None:
                    try:
                        df = file_info['data']
                        df = self._clean_dataframe(df)
                        if main_df is None:
                            main_df = df
                            con.register('main_data', df)
                            logger.info(f"Registered {file_info['filename']} as main_data: {df.shape}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_info['filename']}: {e}")

            # Load data from scraped content
            for scraped in context['scraped_data']:
                if 'tables' in scraped and scraped['tables']:
                    for table_info in scraped['tables']:
                        try:
                            if main_df is None and table_info.get('data'):
                                df = pd.DataFrame(table_info['data'])
                                df = self._clean_dataframe(df)
                                main_df = df
                                con.register('main_data', main_df)
                                logger.info(f"Registered scraped table as main_data: {df.shape}")
                                break
                        except Exception as e:
                            logger.error(f"Error processing scraped table: {e}")

            # Execute plan steps
            for step in plan.get('steps', []):
                step_id = step['step_id']
                action = step['action']
                params = step.get('params', {})

                logger.info(f"Executing {step_id}: {action}")

                try:
                    if action == 'knowledge_response':
                        query = params.get('query', context['questions'])
                        answer = await self.llm.generate_response(f"Answer this question: {query}")
                        results[step_id] = {'type': 'knowledge_response', 'answer': answer}

                    elif action == 'load_data':
                        if main_df is not None:
                            results[step_id] = {'type': 'success', 'message': f'Data loaded: {main_df.shape}'}
                        else:
                            results[step_id] = {'type': 'error', 'error': 'No data available'}

                    elif action == 'run_sql':
                        query = params.get('query', '')
                        if query and main_df is not None:
                            logger.info(f"Executing SQL: {query}")
                            result_df = con.execute(query).df()
                            results[step_id] = {
                                'type': 'dataframe',
                                'data': result_df.to_dict('records'),
                                'columns': list(result_df.columns),
                                'shape': result_df.shape
                            }
                        else:
                            results[step_id] = {'type': 'error', 'error': 'No data or query'}

                    elif action == 'statistical_analysis':
                        if main_df is not None:
                            analysis_result = await self._perform_statistical_analysis(main_df.copy(), params)
                            results[step_id] = analysis_result
                        else:
                            results[step_id] = {'type': 'error', 'error': 'No data for analysis'}

                    elif action == 'create_visualization':
                        if main_df is not None:
                            plot_result = await self._create_visualization(main_df.copy(), params)
                            results[step_id] = plot_result
                        else:
                            results[step_id] = {'type': 'error', 'error': 'No data for visualization'}

                    elif action == 'web_scraping':
                        # Handle web scraping if URLs are in questions
                        urls = re.findall(r'https?://[^\s\)]+', context['questions'])
                        if urls:
                            scraped = await WebScraper.scrape_url(urls[0])
                            results[step_id] = {'type': 'scraped_data', 'data': scraped}
                        else:
                            results[step_id] = {'type': 'error', 'error': 'No URLs found'}

                except Exception as e:
                    logger.error(f"Error in step {step_id}: {e}")
                    results[step_id] = {'type': 'error', 'error': str(e)}

        except Exception as e:
            logger.error(, '', final_response_str)

            parsed_json = json.loads(final_response_str)
            return parsed_json

        except Exception as e:
            logger.error(f"Final answer synthesis failed: {e}")
            return {
                "error": "Failed to generate synthesized response",
                "raw_results": results
            }

# Initialize FastAPI app
app = FastAPI(
    title="Universal Data Analyst Agent",
    description="AI-powered data analysis with Gemini 1.5 Flash",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_llm_provider():
    """Initialize Gemini provider"""
    if not Config.GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    return GeminiProvider(Config.GEMINI_API_KEY)

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Universal Data Analyst Agent</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { font-size: 1.1em; opacity: 0.9; }
            .main-content { padding: 40px; }
            .form-section {
                background: white;
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            }
            .form-group { margin-bottom: 25px; }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #34495e;
                font-size: 1.1em;
            }
            textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #e3e3e3;
                border-radius: 10px;
                font-size: 16px;
                font-family: 'Courier New', monospace;
                resize: vertical;
                transition: border-color 0.3s ease;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            input[type="file"] {
                width: 100%;
                padding: 15px;
                border: 2px dashed #e3e3e3;
                border-radius: 10px;
                background: #f8f9fa;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            input[type="file"]:hover {
                border-color: #667eea;
                background: #f0f2ff;
            }
            .analyze-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                font-size: 1.2em;
                font-weight: 600;
                border-radius: 10px;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 10px;
                margin: 0 auto;
            }
            .analyze-btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            }
            .analyze-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            .spinner {
                width: 20px;
                height: 20px;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-top: 2px solid white;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                display: none;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .result-section {
                margin-top: 30px;
                display: none;
            }
            .result-content {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            }
            .result-data {
                background: #f8f9fa;
                border: 1px solid #e3e3e3;
                border-radius: 10px;
                padding: 20px;
                white-space: pre-wrap;
                font-family: 'Courier New', monospace;
                max-height: 400px;
                overflow-y: auto;
            }
            .error {
                background: #fee;
                border-left: 4px solid #e74c3c;
                color: #c0392b;
            }
            .success {
                border-left: 4px solid #27ae60;
            }
            .status-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-healthy { background: #27ae60; }
            .status-error { background: #e74c3c; }
            .config-info {
                background: #e8f5e8;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                border-left: 4px solid #27ae60;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Universal Data Analyst Agent</h1>
                <p>Intelligent data analysis powered by Gemini 1.5 Flash</p>
            </div>

            <div class="main-content">
                <div id="configInfo" class="config-info">
                    <h3><span class="status-indicator status-healthy"></span>System Status</h3>
                    <p>Loading configuration...</p>
                </div>

                <div class="form-section">
                    <h2>🔍 Analysis Request</h2>
                    
                    <form id="analysisForm" enctype="multipart/form-data">
                        <div class="form-group">
                            <label for="questions">📝 Questions (Required)</label>
                            <textarea id="questions" name="questions" rows="10" 
                                      placeholder="Enter your analysis questions here...

Examples:
1. How many records are in the dataset?
2. What is the correlation between sales and profit?
3. Create a scatter plot of sales vs profit with a regression line.

For web scraping:
Scrape data from https://example.com/data
1. How many rows in the main table?
2. What is the average price?

Response format options:
• JSON Array: [answer1, answer2, ...]
• JSON Object: {'question1': 'answer1', ...}"></textarea>
                        </div>

                        <div class="form-group">
                            <label for="files">📁 Data Files (Optional)</label>
                            <input type="file" id="files" name="files" multiple 
                                   accept=".csv,.xlsx,.xls,.json,.pdf,.png,.jpg,.jpeg,.txt,.tsv">
                            <div style="font-size: 0.9em; color: #666; margin-top: 10px;">
                                Supported: CSV, Excel, JSON, PDF, Images, Text files (Max 50MB each)
                            </div>
                        </div>

                        <div style="text-align: center;">
                            <button type="submit" class="analyze-btn" id="analyzeBtn">
                                <div class="spinner" id="spinner"></div>
                                <span id="btnText">🚀 Start Analysis</span>
                            </button>
                        </div>
                    </form>
                </div>

                <div class="result-section" id="resultSection">
                    <div class="result-content">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                            <h2>📊 Analysis Results</h2>
                            <button onclick="copyResults()" style="background: #3498db; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer;">📋 Copy</button>
                        </div>
                        <div id="resultData" class="result-data"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentApiUrl = window.location.origin + '/api/';
            
            document.addEventListener('DOMContentLoaded', function() {
                loadConfiguration();
                setupForm();
            });

            async function loadConfiguration() {
                try {
                    const response = await fetch('/config');
                    const config = await response.json();
                    
                    const configDiv = document.getElementById('configInfo');
                    configDiv.innerHTML = `
                        <h3><span class="status-indicator status-healthy"></span>System Ready</h3>
                        <p><strong>LLM:</strong> ${config.llm_provider} (${config.llm_model})</p>
                        <p><strong>Max File Size:</strong> ${config.max_file_size_mb}MB</p>
                        <p><strong>API Key:</strong> ${config.has_gemini_key ? '✅ Configured' : '❌ Missing'}</p>
                    `;
                } catch (error) {
                    const configDiv = document.getElementById('configInfo');
                    configDiv.innerHTML = `
                        <h3><span class="status-indicator status-error"></span>Configuration Error</h3>
                        <p>Could not load system configuration</p>
                    `;
                }
            }

            function setupForm() {
                const form = document.getElementById('analysisForm');
                form.addEventListener('submit', handleSubmit);
            }

            async function handleSubmit(e) {
                e.preventDefault();
                
                const questionsInput = document.getElementById('questions');
                const filesInput = document.getElementById('files');
                const analyzeBtn = document.getElementById('analyzeBtn');
                const spinner = document.getElementById('spinner');
                const btnText = document.getElementById('btnText');
                const resultSection = document.getElementById('resultSection');
                
                const questions = questionsInput.value.trim();
                if (!questions) {
                    showError('Please enter some questions to analyze.');
                    return;
                }
                
                // Show loading state
                analyzeBtn.disabled = true;
                spinner.style.display = 'block';
                btnText.textContent = 'Analyzing...';
                resultSection.style.display = 'none';
                
                try {
                    const formData = new FormData();
                    
                    // Add questions as a file
                    const questionsBlob = new Blob([questions], { type: 'text/plain' });
                    formData.append('files', questionsBlob, 'questions.txt');
                    
                    // Add other files
                    Array.from(filesInput.files).forEach(file => {
                        formData.append('files', file);
                    });
                    
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 300000);
                    
                    const response = await fetch(currentApiUrl, {
                        method: 'POST',
                        body: formData,
                        signal: controller.signal
                    });
                    
                    clearTimeout(timeoutId);
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(`HTTP ${response.status}: ${errorText}`);
                    }
                    
                    const result = await response.json();
                    showResult(result);
                    
                } catch (error) {
                    console.error('Analysis failed:', error);
                    let errorMessage = 'Analysis failed: ';
                    
                    if (error.name === 'AbortError') {
                        errorMessage += 'Request timed out';
                    } else {
                        errorMessage += error.message;
                    }
                    
                    showError(errorMessage);
                } finally {
                    // Reset button state
                    analyzeBtn.disabled = false;
                    spinner.style.display = 'none';
                    btnText.textContent = '🚀 Start Analysis';
                }
            }

            function showResult(result) {
                const resultSection = document.getElementById('resultSection');
                const resultData = document.getElementById('resultData');
                
                let displayText;
                try {
                    displayText = JSON.stringify(result, null, 2);
                } catch (error) {
                    displayText = String(result);
                }
                
                resultData.innerHTML = displayText;
                resultData.className = 'result-data success';
                resultSection.style.display = 'block';
                resultSection.scrollIntoView({ behavior: 'smooth' });
                
                window.lastResult = result;
            }

            function showError(message) {
                const resultSection = document.getElementById('resultSection');
                const resultData = document.getElementById('resultData');
                
                resultData.innerHTML = message;
                resultData.className = 'result-data error';
                resultSection.style.display = 'block';
                resultSection.scrollIntoView({ behavior: 'smooth' });
            }

            function copyResults() {
                if (window.lastResult) {
                    const textToCopy = typeof window.lastResult === 'object' 
                        ? JSON.stringify(window.lastResult, null, 2)
                        : String(window.lastResult);
                    
                    navigator.clipboard.writeText(textToCopy).then(() => {
                        alert('Results copied to clipboard!');
                    }).catch(() => {
                        const textArea = document.createElement('textarea');
                        textArea.value = textToCopy;
                        document.body.appendChild(textArea);
                        textArea.select();
                        document.execCommand('copy');
                        document.body.removeChild(textArea);
                        alert('Results copied to clipboard!');
                    });
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "llm_provider": "gemini",
        "model": "gemini-1.5-flash",
        "version": "2.0.0"
    }

@app.get("/config")
async def get_config():
    """Get system configuration"""
    return {
        "llm_provider": "gemini",
        "llm_model": "gemini-1.5-flash",
        "max_file_size_mb": Config.MAX_FILE_SIZE // (1024 * 1024),
        "request_timeout_seconds": Config.REQUEST_TIMEOUT,
        "has_gemini_key": bool(Config.GEMINI_API_KEY),
        "supported_file_types": [
            "csv", "tsv", "xlsx", "xls", "json", "pdf", "txt", "md",
            "png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"
        ]
    }

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(default=[]), questions: str = Form(default="")):
    """Main analysis endpoint"""
    start_time = datetime.now()

    try:
        # Extract questions from different sources
        questions_content = questions
        data_files = []

        # Check if questions come from uploaded files
        for file in files:
            if file.filename == "questions.txt":
                questions_content = (await file.read()).decode('utf-8')
            else:
                await file.seek(0)
                data_files.append(file)

        if not questions_content.strip():
            raise HTTPException(status_code=400, detail="No questions provided")

        # Validate file sizes
        for file in data_files:
            if hasattr(file, 'size') and file.size and file.size > Config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {file.filename} exceeds maximum size of {Config.MAX_FILE_SIZE // (1024*1024)}MB"
                )

        # Initialize agent and analyze
        llm_provider = get_llm_provider()
        agent = DataAnalystAgent(llm_provider)

        logger.info(f"Starting analysis with {len(data_files)} files")
        logger.info(f"Questions: {questions_content[:200]}...")

        result = await agent.analyze(questions_content, data_files if data_files else None)

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Error handlers
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Development server runner
if __name__ == "__main__":
    print("\n🚀 Starting Universal Data Analyst Agent...")
    print(f"🤖 LLM Provider: Gemini 1.5 Flash")
    print(f"📁 Max File Size: {Config.MAX_FILE_SIZE // (1024*1024)}MB")
    print(f"⏱️  Request Timeout: {Config.REQUEST_TIMEOUT}s")
    print(f"🔑 Gemini API Key: {'✅ Configured' if Config.GEMINI_API_KEY else '❌ Missing'}")
    print("\n🌐 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")

    if not Config.GEMINI_API_KEY:
        print("\n⚠️  WARNING: GEMINI_API_KEY not set!")
        print("   Set it with: export GEMINI_API_KEY='your-api-key-here'")
        print("   Or create a .env file with: GEMINI_API_KEY=your-api-key-here")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=True
    )f"Gemini API error: {e}")
            raise

class WebScraper:
    """Handles web scraping with various strategies"""

    @staticmethod
    async def scrape_url(url: str) -> Dict[str, Any]:
        """Scrape data from URL"""
        try:
            # Clean URL first - remove any trailing characters
            url = url.rstrip(')').rstrip('.')

            async with httpx.AsyncClient(timeout=60.0) as client:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = await client.get(url, follow_redirects=True, headers=headers)
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

            # Get file extension and MIME type
            file_ext = Path(filename).suffix.lower()
            mime_type, _ = mimetypes.guess_type(filename)

            # Handle various file types
            if file_ext in ['.csv', '.tsv'] or (mime_type and 'csv' in mime_type):
                file_info["type"] = "tabular"
                # Try different encodings and separators
                encodings = ['utf-8', 'latin-1', 'cp1252']
                separators = [',', '\t', ';', '|']
                
                for encoding in encodings:
                    for sep in separators:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding, sep=sep, on_bad_lines='skip')
                            if df.shape[1] > 1:  # Valid table
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

            elif file_ext in ['.xlsx', '.xls'] or (mime_type and 'spreadsheet' in mime_type):
                file_info["type"] = "tabular"
                try:
                    df = pd.read_excel(file_path)
                    file_info["data"] = df
                    file_info["metadata"] = {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                    }
                except Exception as e:
                    logger.error(f"Excel processing error: {e}")

            elif file_ext == '.json' or (mime_type and 'json' in mime_type):
                file_info["type"] = "json"
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    file_info["data"] = data
                    file_info["metadata"] = {"keys": list(data.keys()) if isinstance(data, dict) else "array"}
                except Exception:
                    # Try with different encodings
                    for encoding in ['latin-1', 'cp1252']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                data = json.load(f)
                            file_info["data"] = data
                            file_info["metadata"] = {"keys": list(data.keys()) if isinstance(data, dict) else "array"}
                            break
                        except Exception:
                            continue

            elif file_ext == '.pdf' or (mime_type and 'pdf' in mime_type):
                file_info["type"] = "text"
                text = await FileProcessor._extract_pdf_text(file_path)
                file_info["data"] = text
                file_info["metadata"] = {"length": len(text), "pages": text.count('\n\n') + 1}

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

            elif file_ext in ['.txt', '.md', '.rst'] or (mime_type and 'text' in mime_type):
                # Try to read as a table first, then as plain text
                encodings = ['utf-8', 'latin-1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        # First attempt: structured data
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
                
                # If not tabular, read as plain text
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

            else:
                # Unknown file type - try to process as text
                file_info["type"] = "unknown"
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read(1000)  # Read first 1000 chars
                    if text.isprintable():
                        file_info["type"] = "text"
                        file_info["data"] = text
                        file_info["metadata"] = {"note": "Unknown text file type", "preview_length": len(text)}
                except Exception:
                    with open(file_path, 'rb') as f:
                        data = f.read(100)  # Read first 100 bytes
                    file_info["metadata"] = {"note": "Binary file - not processed", "size_bytes": os.path.getsize(file_path)}

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

    async def analyze(self, questions: str, files: List[UploadFile] = None) -> Any:
        """Main analysis pipeline"""
        try:
            # Step 1: Process all files (if any)
            processed_files = []
            if files:
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
                "questions": questions,
                "has_data": bool(processed_files or scraped_data)
            }

            # Step 4: Generate execution plan
            plan = await self._create_execution_plan(context)

            # Step 5: Execute the plan
            execution_results = await self._execute_plan(plan, context)

            # Step 6: Generate final response
            final_answer = await self._generate_final_answer(questions, execution_results, context, plan)

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

        data_summary = ""
        if context['files']:
            data_summary += f"Files: {len(context['files'])} files\n"
            for f in context['files']:
                data_summary += f"  - {f.get('filename', 'unknown')} ({f.get('type', 'unknown')})\n"
        
        if context['scraped_data']:
            data_summary += f"Scraped URLs: {len(context['scraped_data'])}\n"

        plan_prompt = f"""You are an expert data analyst. Create a comprehensive execution plan to answer the user's questions.

QUESTIONS TO ANSWER:
{context['questions']}

AVAILABLE DATA:
{data_summary}

DATA AVAILABLE: {context['has_data']}

Create a JSON execution plan. Use 'main_data' as the primary table name for SQL queries.

Response format (JSON only, no markdown):
{{
    "analysis_type": "knowledge_based|data_analysis|web_scraping|mixed",
    "expected_response_format": "json_array|json_object|text",
    "steps": [
        {{
            "step_id": "step_1",
            "action": "knowledge_response|load_data|run_sql|statistical_analysis|create_visualization|web_scraping",
            "description": "what this step does",
            "params": {{
                "query": "SQL query or question",
                "plot_type": "scatter|bar|line",
                "x_col": "column_name",
                "y_col": "column_name",
                "analysis_type": "correlation|regression"
            }}
        }}
    ]
}}"""

        try:
            plan_text = await self.llm.generate_response(plan_prompt, json_mode=True)
            # Clean up response
            plan_text = re.sub(r'^```json\s*', '', plan_text)
            plan_text = re.sub(r'\s*```$', '', plan_text)

            plan = json.loads(plan_text)
            logger.info(f"Generated execution plan: {json.dumps(plan, indent=2)}")
            return plan
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            # Fallback plan
            return {
                "analysis_type": "knowledge_based" if not context['has_data'] else "data_analysis",
                "expected_response_format": "json_object",
                "steps": [
                    {
                        "step_id": "step_1", 
                        "action": "knowledge_response" if not context['has_data'] else "load_data", 
                        "description": "Provide answer based on available information", 
                        "params": {"query": context['questions']}
                    }
                ]
            }

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataframe"""
        # Clean column names
        df.columns = [re.sub(r'[^A-Za-z0-9_]', '', str(col).strip().replace(' ', '_').replace('(', '').replace(')', '')) for col in df.columns]

        for col in df.columns:
            # Attempt to convert to datetime for date-like columns
            if df[col].dtype == 'object':
                if any(term in col.lower() for term in ['date', 'time', 'year']):
                    try:
                        converted_dates = pd.to_datetime(df[col], errors='coerce')
                        if converted_dates.notna().sum() / len(df[col]) > 0.5:
                            df[col] = converted_dates
                            continue
                    except Exception:
                        pass

            # Clean and convert numeric columns
            if df[col].dtype == 'object':
                try:
                    cleaned_series = df[col].astype(str).str.replace(r'[$,%\s]', '', regex=True)
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    if numeric_series.notna().sum() / len(df[col]) > 0.7:
                        df[col] = numeric_series
                except Exception:
                    pass

        return df

    async def _perform_statistical_analysis(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Perform statistical analysis"""
        try:
            analysis_type = params.get('analysis_type', 'correlation')
            columns = params.get('columns', [])

            # Convert datetime columns to numeric for analysis
            numeric_df = df.copy()
            for col in numeric_df.columns:
                if pd.api.types.is_datetime64_any_dtype(numeric_df[col]):
                    numeric_df[col] = numeric_df[col].apply(lambda x: x.toordinal() if pd.notna(x) else np.nan)

            if analysis_type == 'correlation':
                if len(columns) >= 2:
                    numeric_cols = [col for col in columns if col in numeric_df.select_dtypes(include=np.number).columns]
                else:
                    numeric_cols = numeric_df.select_dtypes(include=[np.number]).columns.tolist()

                if len(numeric_cols) >= 2:
                    corr_matrix = numeric_df[numeric_cols].corr()
                    corr_values = corr_matrix.abs().unstack()
                    corr_values = corr_values[corr_values < 1.0]
                    
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

            elif analysis_type == 'regression' and len(columns) >= 2:
                x_col, y_col = columns[0], columns[1]

                if x_col in numeric_df.columns and y_col in numeric_df.columns:
                    clean_data = numeric_df[[x_col, y_col]].dropna()
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

            return {'type': 'statistical_analysis', 'error': f'Cannot perform {analysis_type} analysis.'}

        except Exception as e:
            logger.error(f"Statistical analysis error: {e}")
            return {'type': 'statistical_analysis', 'error': str(e)}

    async def _create_visualization(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Create visualizations"""
        try:
            plot_type = params.get('plot_type', 'bar')
            x_col = params.get('x_col') or params.get('x_axis')
            y_col = params.get('y_col') or params.get('y_axis')
            
            if not x_col:
                cat_cols = df.select_dtypes(include=['object', 'category']).columns
                x_col = cat_cols[0] if len(cat_cols) > 0 else df.columns[0]
            
            if not y_col and plot_type != 'bar':
                num_cols = df.select_dtypes(include=np.number).columns
                y_col = num_cols[0] if len(num_cols) > 0 else None

            if not x_col or x_col not in df.columns:
                return {'type': 'visualization', 'error': f'Column "{x_col}" not found. Available: {list(df.columns)}'}

            # Create plot
            plt.figure(figsize=(12, 8))
            plt.style.use('default')

            plot_data = df.copy()
            plot_cols = [c for c in [x_col, y_col] if c is not None]
            plot_data.dropna(subset=plot_cols, inplace=True)
            
            if plot_data.empty:
                return {'type': 'visualization', 'error': 'No data available for plotting after removing missing values.'}

            if plot_type == 'scatter' and y_col:
                sns.scatterplot(x=x_col, y=y_col, data=plot_data, alpha=0.7)
                if params.get('add_regression', False):
                    sns.regplot(x=x_col, y=y_col, data=plot_data, scatter=False, color='red', linestyle='--')
            elif plot_type == 'bar':
                if y_col:
                    plot_values = plot_data.groupby(x_col)[y_col].sum().sort_values(ascending=False)
                    sns.barplot(x=plot_values.index, y=plot_values.values, palette="viridis")
                else:
                    sns.countplot(x=x_col, data=plot_data, palette="viridis")
            elif plot_type == 'line' and y_col:
                sns.lineplot(x=x_col, y=y_col, data=plot_data, marker='o')

            plt.xlabel(str(x_col).replace('_', ' ').title(), fontsize=14)
            plt.ylabel(str(y_col if y_col else 'Count').replace('_', ' ').title(), fontsize=14)
            plt.xticks(rotation=45, ha='right')
            
            title_y = y_col if y_col else 'Count'
            title = f"{plot_type.title()} of {str(title_y).replace('_',' ').title()} by {str(x_col).replace('_',' ').title()}"
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()

            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            img_data = base64.b64encode(buffer.getvalue()).decode()

            return {
                'type': 'visualization',
                'format': 'base64_png',
                'data': f"data:image/png;base64,{img_data}",
                'size_bytes': len(img_data),
                'plot_info': {
                    'type': plot_type,
                    'x_column': x_col,
                    'y_column': y_col,
                    'title': title
                }
            }

        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return {'type': 'visualization', 'error': str(e)}

    async def _execute_plan(self, plan: Dict, context: Dict) -> Dict:
        """Execute the analysis plan"""
        results = {}

        # Initialize DuckDB connection
        con = duckdb.connect(database=':memory:', read_only=False)
        try:
            main_df = None

            # Load data from files
            for file_info in context['files']:
                if file_info.get('type') == 'tabular' and file_info.get('data') is not None:
                    try:
                        df = file_info['data']
                        df = self._clean_dataframe(df)
                        if main_df is None:
                            main_df = df
                            con.register('main_data', df)
                            logger.info(f"Registered {file_info['filename']} as main_data: {df.shape}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_info['filename']}: {e}")

            # Load data from scraped content
            for scraped in context['scraped_data']:
                if 'tables' in scraped and scraped['tables']:
                    for table_info in scraped['tables']:
                        try:
                            if main_df is None and table_info.get('data'):
                                df = pd.DataFrame(table_info['data'])
                                df = self._clean_dataframe(df)
                                main_df = df
                                con.register('main_data', main_df)
                                logger.info(f"Registered scraped table as main_data: {df.shape}")
                                break
                        except Exception as e:
                            logger.error(f"Error processing scraped table: {e}")

            # Execute plan steps
            for step in plan.get('steps', []):
                step_id = step['step_id']
                action = step['action']
                params = step.get('params', {})

                logger.info(f"Executing {step_id}: {action}")

                try:
                    if action == 'knowledge_response':
                        query = params.get('query', context['questions'])
                        answer = await self.llm.generate_response(f"Answer this question: {query}")
                        results[step_id] = {'type': 'knowledge_response', 'answer': answer}

                    elif action == 'load_data':
                        if main_df is not None:
                            results[step_id] = {'type': 'success', 'message': f'Data loaded: {main_df.shape}'}
                        else:
                            results[step_id] = {'type': 'error', 'error': 'No data available'}

                    elif action == 'run_sql':
                        query = params.get('query', '')
                        if query and main_df is not None:
                            logger.info(f"Executing SQL: {query}")
                            result_df = con.execute(query).df()
                            results[step_id] = {
                                'type': 'dataframe',
                                'data': result_df.to_dict('records'),
                                'columns': list(result_df.columns),
                                'shape': result_df.shape
                            }
                        else:
                            results[step_id] = {'type': 'error', 'error': 'No data or query'}

                    elif action == 'statistical_analysis':
                        if main_df is not None:
                            analysis_result = await self._perform_statistical_analysis(main_df.copy(), params)
                            results[step_id] = analysis_result
                        else:
                            results[step_id] = {'type': 'error', 'error': 'No data for analysis'}

                    elif action == 'create_visualization':
                        if main_df is not None:
                            plot_result = await self._create_visualization(main_df.copy(), params)
                            results[step_id] = plot_result
                        else:
                            results[step_id] = {'type': 'error', 'error': 'No data for visualization'}

                    elif action == 'web_scraping':
                        # Handle web scraping if URLs are in questions
                        urls = re.findall(r'https?://[^\s\)]+', context['questions'])
                        if urls:
                            scraped = await WebScraper.scrape_url(urls[0])
                            results[step_id] = {'type': 'scraped_data', 'data': scraped}
                        else:
                            results[step_id] = {'type': 'error', 'error': 'No URLs found'}

                except Exception as e:
                    logger.error(f"Error in step {step_id}: {e}")
                    results[step_id] = {'type': 'error', 'error': str(e)}

        except Exception as e:
            logger.error( f"Execution plan failed: {e}")
            raise HTTPException(status_code=500, detail=f"Execution plan failed: {str(e)}")
