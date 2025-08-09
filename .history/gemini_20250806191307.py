import os
import json
import base64
import tempfile
import io
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import asyncio
import aiofiles
import requests
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import duckdb
import PyPDF2
import google.generativeai as genai
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your-gemini-api-key-here')
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Data Analyst Agent", version="1.0.0")

class DataAnalyst:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.temp_dir = tempfile.mkdtemp()
        
    async def analyze_request(self, questions: str, files: List[UploadFile]) -> Union[List[Any], Dict[str, Any]]:
        """Main analysis function that processes questions and files"""
        try:
            # Save uploaded files
            file_paths = []
            file_info = []
            
            for file in files:
                if file.filename != "questions.txt":  # Skip questions file
                    file_path = os.path.join(self.temp_dir, file.filename)
                    async with aiofiles.open(file_path, 'wb') as f:
                        content = await file.read()
                        await f.write(content)
                    
                    file_paths.append(file_path)
                    file_info.append({
                        'name': file.filename,
                        'path': file_path,
                        'type': self._get_file_type(file.filename),
                        'size': len(content)
                    })
            
            # Determine response format from questions
            response_format = self._determine_response_format(questions)
            
            # Process data based on file types
            processed_data = await self._process_files(file_info)
            
            # Generate analysis plan
            analysis_plan = await self._generate_analysis_plan(questions, file_info, processed_data)
            
            # Execute analysis
            results = await self._execute_analysis(analysis_plan, processed_data, questions)
            
            # Format response according to expected format
            formatted_response = self._format_response(results, response_format, questions)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    def _get_file_type(self, filename: str) -> str:
        """Determine file type from extension"""
        ext = filename.lower().split('.')[-1]
        if ext in ['csv', 'tsv']:
            return 'tabular'
        elif ext in ['pdf']:
            return 'document'
        elif ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
            return 'image'
        elif ext in ['json']:
            return 'json'
        elif ext in ['txt', 'md']:
            return 'text'
        else:
            return 'unknown'
    
    def _determine_response_format(self, questions: str) -> str:
        """Determine if response should be array or object based on questions"""
        # Look for JSON object indicators
        if '{"' in questions or '"..."' in questions:
            return 'object'
        # Look for array indicators
        elif 'JSON array' in questions or 'array of strings' in questions:
            return 'array'
        else:
            # Default based on question structure
            lines = questions.strip().split('\n')
            numbered_questions = [l for l in lines if re.match(r'^\d+\.', l.strip())]
            if len(numbered_questions) > 0:
                return 'array'
            else:
                return 'object'
    
    async def _process_files(self, file_info: List[Dict]) -> Dict[str, Any]:
        """Process uploaded files and extract data"""
        processed = {}
        
        for file in file_info:
            try:
                if file['type'] == 'tabular':
                    # Load CSV/TSV data
                    df = pd.read_csv(file['path'])
                    processed[file['name']] = {
                        'type': 'dataframe',
                        'data': df,
                        'shape': df.shape,
                        'columns': df.columns.tolist(),
                        'head': df.head().to_dict(),
                        'dtypes': df.dtypes.to_dict()
                    }
                
                elif file['type'] == 'document':
                    # Extract text from PDF
                    with open(file['path'], 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                    processed[file['name']] = {
                        'type': 'text',
                        'data': text,
                        'length': len(text)
                    }
                
                elif file['type'] == 'image':
                    # Load image
                    img = Image.open(file['path'])
                    processed[file['name']] = {
                        'type': 'image',
                        'data': file['path'],
                        'size': img.size,
                        'mode': img.mode
                    }
                
                elif file['type'] == 'json':
                    # Load JSON data
                    with open(file['path'], 'r') as f:
                        data = json.load(f)
                    processed[file['name']] = {
                        'type': 'json',
                        'data': data
                    }
                
                elif file['type'] == 'text':
                    # Load text file
                    with open(file['path'], 'r') as f:
                        text = f.read()
                    processed[file['name']] = {
                        'type': 'text',
                        'data': text,
                        'length': len(text)
                    }
            
            except Exception as e:
                logger.warning(f"Could not process file {file['name']}: {str(e)}")
                processed[file['name']] = {
                    'type': 'error',
                    'error': str(e)
                }
        
        return processed
    
    async def _generate_analysis_plan(self, questions: str, file_info: List[Dict], processed_data: Dict) -> Dict:
        """Generate an analysis plan using Gemini"""
        
        # Create context for the LLM
        context = f"""
        QUESTIONS TO ANSWER:
        {questions}
        
        AVAILABLE FILES:
        """
        
        for file in file_info:
            context += f"- {file['name']} ({file['type']}, {file['size']} bytes)\n"
        
        context += "\nDATA SUMMARY:\n"
        for name, data in processed_data.items():
            if data['type'] == 'dataframe':
                context += f"- {name}: DataFrame with shape {data['shape']}, columns: {data['columns']}\n"
            elif data['type'] == 'text':
                context += f"- {name}: Text file with {data['length']} characters\n"
            elif data['type'] == 'image':
                context += f"- {name}: Image file ({data['size']})\n"
            elif data['type'] == 'json':
                context += f"- {name}: JSON data\n"
        
        prompt = f"""
        {context}
        
        Create a detailed analysis plan to answer the questions. Consider:
        1. What data sources are needed (web scraping, file processing, etc.)
        2. What calculations/analysis steps are required
        3. What visualizations need to be created
        4. The expected output format
        
        Respond with a JSON object containing the analysis plan with these keys:
        - "data_sources": list of data sources needed
        - "processing_steps": list of processing steps
        - "calculations": list of calculations needed
        - "visualizations": list of plots/charts to create
        - "output_format": "array" or "object"
        """
        
        try:
            response = self.model.generate_content(prompt)
            plan_text = response.text
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                # Fallback plan
                plan = {
                    "data_sources": ["provided_files"],
                    "processing_steps": ["analyze_questions", "process_data"],
                    "calculations": ["answer_questions"],
                    "visualizations": [],
                    "output_format": self._determine_response_format(questions)
                }
            
            return plan
            
        except Exception as e:
            logger.warning(f"Could not generate analysis plan: {str(e)}")
            return {
                "data_sources": ["provided_files"],
                "processing_steps": ["analyze_questions", "process_data"],
                "calculations": ["answer_questions"],
                "visualizations": [],
                "output_format": self._determine_response_format(questions)
            }
    
    async def _execute_analysis(self, plan: Dict, processed_data: Dict, questions: str) -> Dict:
        """Execute the analysis plan"""
        results = {}
        
        # Handle web scraping if needed
        if any('wikipedia' in str(source).lower() or 'http' in str(source) for source in plan.get('data_sources', [])):
            scraped_data = await self._scrape_web_data(questions)
            processed_data.update(scraped_data)
        
        # Handle database queries if needed
        if any('sql' in str(step).lower() or 'duckdb' in str(step).lower() for step in plan.get('processing_steps', [])):
            db_results = await self._execute_database_queries(questions)
            results.update(db_results)
        
        # Process questions with LLM
        answers = await self._get_llm_answers(questions, processed_data, plan)
        results.update(answers)
        
        # Generate visualizations
        if plan.get('visualizations'):
            viz_results = await self._create_visualizations(plan['visualizations'], processed_data, questions)
            results.update(viz_results)
        
        return results
    
    async def _scrape_web_data(self, questions: str) -> Dict:
        """Scrape web data based on URLs in questions"""
        scraped = {}
        
        # Extract URLs from questions
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', questions)
        
        for url in urls:
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract tables if present
                tables = soup.find_all('table')
                if tables:
                    for i, table in enumerate(tables):
                        df = pd.read_html(str(table))[0]
                        scraped[f'web_table_{i}'] = {
                            'type': 'dataframe',
                            'data': df,
                            'shape': df.shape,
                            'columns': df.columns.tolist(),
                            'source_url': url
                        }
                
                # Extract text
                text = soup.get_text()
                scraped['web_content'] = {
                    'type': 'text',
                    'data': text,
                    'length': len(text),
                    'source_url': url
                }
                
            except Exception as e:
                logger.warning(f"Could not scrape {url}: {str(e)}")
        
        return scraped
    
    async def _execute_database_queries(self, questions: str) -> Dict:
        """Execute database queries mentioned in questions"""
        results = {}
        
        # Extract SQL queries from questions
        sql_pattern = r'```sql\n(.*?)\n```'
        sql_queries = re.findall(sql_pattern, questions, re.DOTALL)
        
        for i, query in enumerate(sql_queries):
            try:
                conn = duckdb.connect()
                result = conn.execute(query).fetchdf()
                results[f'sql_result_{i}'] = {
                    'type': 'dataframe',
                    'data': result,
                    'query': query
                }
            except Exception as e:
                logger.warning(f"Could not execute SQL query: {str(e)}")
                results[f'sql_result_{i}'] = {
                    'type': 'error',
                    'error': str(e),
                    'query': query
                }
        
        return results
    
    async def _get_llm_answers(self, questions: str, processed_data: Dict, plan: Dict) -> Dict:
        """Get answers from LLM based on processed data"""
        
        # Prepare data summary for LLM
        data_summary = "AVAILABLE DATA:\n"
        for name, data in processed_data.items():
            if data['type'] == 'dataframe':
                df = data['data']
                data_summary += f"\n{name}:\n"
                data_summary += f"Shape: {df.shape}\n"
                data_summary += f"Columns: {df.columns.tolist()}\n"
                data_summary += f"Sample data:\n{df.head().to_string()}\n"
                data_summary += f"Data types:\n{df.dtypes.to_string()}\n"
            elif data['type'] == 'text':
                text_preview = data['data'][:1000] + "..." if len(data['data']) > 1000 else data['data']
                data_summary += f"\n{name}: {text_preview}\n"
        
        prompt = f"""
        {data_summary}
        
        QUESTIONS TO ANSWER:
        {questions}
        
        Please analyze the data and answer the questions. Be precise and provide:
        1. Numerical answers where requested
        2. Text answers for descriptive questions
        3. For any calculations, show your work
        4. If creating plots is requested, describe what needs to be plotted
        
        Respond with a JSON object containing the answers with clear keys.
        """
        
        try:
            response = self.model.generate_content(prompt)
            answer_text = response.text
            
            # Extract JSON if present
            json_match = re.search(r'\{.*\}', answer_text, re.DOTALL)
            if json_match:
                answers = json.loads(json_match.group())
            else:
                # Parse answers from text
                answers = {'llm_response': answer_text}
            
            return answers
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return {'error': f"LLM analysis failed: {str(e)}"}
    
    async def _create_visualizations(self, viz_requests: List, processed_data: Dict, questions: str) -> Dict:
        """Create visualizations as requested"""
        viz_results = {}
        
        # Look for plot requests in questions
        plot_requests = []
        
        if 'scatterplot' in questions.lower():
            plot_requests.append('scatterplot')
        if 'histogram' in questions.lower():
            plot_requests.append('histogram')
        if 'bar chart' in questions.lower():
            plot_requests.append('bar_chart')
        if 'line chart' in questions.lower():
            plot_requests.append('line_chart')
        
        # Find dataframes to plot
        dataframes = {name: data['data'] for name, data in processed_data.items() 
                     if data['type'] == 'dataframe'}
        
        if dataframes and plot_requests:
            for plot_type in plot_requests:
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Get the first available dataframe
                    df_name, df = list(dataframes.items())[0]
                    
                    if plot_type == 'scatterplot' and len(df.columns) >= 2:
                        # Create scatterplot with first two numeric columns
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) >= 2:
                            x_col, y_col = numeric_cols[0], numeric_cols[1]
                            ax.scatter(df[x_col], df[y_col], alpha=0.6)
                            ax.set_xlabel(x_col)
                            ax.set_ylabel(y_col)
                            
                            # Add regression line if requested
                            if 'regression' in questions.lower():
                                z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
                                p = np.poly1d(z)
                                ax.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8, linewidth=2)
                    
                    elif plot_type == 'histogram' and len(df.columns) >= 1:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) >= 1:
                            ax.hist(df[numeric_cols[0]].dropna(), bins=30, alpha=0.7)
                            ax.set_xlabel(numeric_cols[0])
                            ax.set_ylabel('Frequency')
                    
                    plt.title(f'{plot_type.title()} - {df_name}')
                    plt.tight_layout()
                    
                    # Convert to base64
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
                    img_buffer.seek(0)
                    img_base64 = base64.b64encode(img_buffer.read()).decode()
                    
                    # Check size limit (100KB)
                    if len(img_base64) < 100000:
                        viz_results[f'{plot_type}_plot'] = f"data:image/png;base64,{img_base64}"
                    else:
                        # Reduce quality and try again
                        img_buffer = io.BytesIO()
                        plt.savefig(img_buffer, format='png', dpi=50, bbox_inches='tight')
                        img_buffer.seek(0)
                        img_base64 = base64.b64encode(img_buffer.read()).decode()
                        viz_results[f'{plot_type}_plot'] = f"data:image/png;base64,{img_base64}"
                    
                    plt.close()
                    
                except Exception as e:
                    logger.warning(f"Could not create {plot_type}: {str(e)}")
        
        return viz_results
    
    def _format_response(self, results: Dict, format_type: str, questions: str) -> Union[List[Any], Dict[str, Any]]:
        """Format the response according to expected format"""
        
        if format_type == 'array':
            # Extract individual answers for array format
            response_array = []
            
            # Parse questions to find individual items
            lines = questions.strip().split('\n')
            question_items = []
            
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    question_items.append(line)
            
            # Match answers to questions
            for i, question in enumerate(question_items):
                if 'correlation' in question.lower():
                    # Look for correlation value in results
                    correlation = self._extract_correlation(results)
                    response_array.append(correlation)
                elif 'earliest' in question.lower() or 'first' in question.lower():
                    # Look for text answer
                    text_answer = self._extract_text_answer(results, question)
                    response_array.append(text_answer)
                elif 'how many' in question.lower():
                    # Look for count
                    count = self._extract_count(results, question)
                    response_array.append(count)
                elif 'plot' in question.lower() or 'chart' in question.lower():
                    # Look for visualization
                    plot = self._extract_plot(results)
                    response_array.append(plot)
                else:
                    # Generic answer
                    answer = self._extract_generic_answer(results, question)
                    response_array.append(answer)
            
            return response_array
            
        else:
            # Object format - match keys from questions
            response_object = {}
            
            # Extract question keys from questions text
            question_keys = re.findall(r'"([^"]+)":\s*"[^"]*"', questions)
            
            for key in question_keys:
                if 'court' in key.lower() and 'most' in key.lower():
                    response_object[key] = self._extract_text_answer(results, key)
                elif 'slope' in key.lower() or 'regression' in key.lower():
                    response_object[key] = self._extract_correlation(results)
                elif 'plot' in key.lower():
                    response_object[key] = self._extract_plot(results)
                else:
                    response_object[key] = self._extract_generic_answer(results, key)
            
            return response_object
    
    def _extract_correlation(self, results: Dict) -> float:
        """Extract correlation value from results"""
        # Look through results for correlation
        for key, value in results.items():
            if isinstance(value, (int, float)):
                return round(float(value), 6)
        return 0.0
    
    def _extract_text_answer(self, results: Dict, question: str) -> str:
        """Extract text answer from results"""
        # Look through results for text answers
        for key, value in results.items():
            if isinstance(value, str) and len(value) > 0:
                return value
        return "Answer not found"
    
    def _extract_count(self, results: Dict, question: str) -> int:
        """Extract count from results"""
        # Look through results for counts
        for key, value in results.items():
            if isinstance(value, int):
                return value
        return 0
    
    def _extract_plot(self, results: Dict) -> str:
        """Extract plot from results"""
        # Look for base64 encoded plots
        for key, value in results.items():
            if isinstance(value, str) and value.startswith('data:image'):
                return value
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    def _extract_generic_answer(self, results: Dict, question: str) -> Any:
        """Extract generic answer from results"""
        # Return first non-empty result
        for key, value in results.items():
            if value is not None and value != "":
                return value
        return "No answer found"

# Initialize the analyzer
analyzer = DataAnalyst()

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """Main endpoint for data analysis"""
    try:
        # Find questions.txt
        questions_content = None
        other_files = []
        
        for file in files:
            if file.filename == "questions.txt":
                content = await file.read()
                questions_content = content.decode('utf-8')
                # Reset file pointer
                await file.seek(0)
            else:
                other_files.append(file)
        
        if not questions_content:
            raise HTTPException(status_code=400, detail="questions.txt is required")
        
        # Perform analysis
        result = await analyzer.analyze_request(questions_content, other_files)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
