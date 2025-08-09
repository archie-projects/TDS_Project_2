import os
import json
import base64
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import tempfile
import io

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import sqlite3
import duckdb
from PIL import Image
import PyPDF2
import openai
from openai import OpenAI
import re
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Data Analyst Agent", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure AI Pipe client
client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN", "your-aipipe-token-here"),
    base_url="https://aipipe.org/openai/v1"
)

class DataAnalystAgent:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        
    async def analyze_task(self, questions: str, files: List[UploadFile] = None) -> Any:
        """Main analysis function that handles any data analysis task"""
        try:
            # Prepare context for LLM
            context = {
                "questions": questions,
                "files_info": [],
                "available_data": {}
            }
            
            # Process uploaded files
            if files:
                for file in files:
                    file_info = await self._process_file(file)
                    context["files_info"].append(file_info)
                    context["available_data"][file.filename] = file_info
            
            # Generate analysis plan using LLM
            analysis_plan = await self._generate_analysis_plan(context)
            
            # Execute the analysis plan
            results = await self._execute_analysis_plan(analysis_plan, context)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in analyze_task: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _process_file(self, file: UploadFile) -> Dict[str, Any]:
        """Process uploaded files and extract information"""
        content = await file.read()
        file_info = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "content": None,
            "summary": None
        }
        
        try:
            if file.content_type == "text/plain":
                file_info["content"] = content.decode('utf-8')
                file_info["summary"] = "Text file content"
                
            elif file.content_type == "text/csv" or file.filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                file_info["content"] = df.to_dict('records')
                file_info["summary"] = f"CSV with {len(df)} rows and {len(df.columns)} columns. Columns: {list(df.columns)}"
                file_info["dataframe"] = df
                
            elif file.content_type == "application/json" or file.filename.endswith('.json'):
                file_info["content"] = json.loads(content.decode('utf-8'))
                file_info["summary"] = "JSON data file"
                
            elif file.content_type.startswith('image/'):
                # Convert image to base64 for LLM vision
                file_info["content"] = base64.b64encode(content).decode('utf-8')
                file_info["summary"] = f"Image file ({file.content_type})"
                
            elif file.content_type == "application/pdf" or file.filename.endswith('.pdf'):
                # Extract text from PDF
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                file_info["content"] = text
                file_info["summary"] = f"PDF with {len(pdf_reader.pages)} pages"
                
            else:
                file_info["summary"] = "Binary file - content not processed"
                
        except Exception as e:
            file_info["summary"] = f"Error processing file: {str(e)}"
            logger.error(f"Error processing file {file.filename}: {str(e)}")
        
        return file_info
    
    async def _generate_analysis_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis plan using LLM"""
        
        system_prompt = """You are a data analyst agent. Given a task description and available data, create a comprehensive analysis plan.

You have access to the following capabilities:
1. Web scraping (requests, BeautifulSoup)
2. Data processing (pandas, numpy)
3. Database queries (sqlite3, duckdb)
4. Visualization (matplotlib, seaborn)
5. Statistical analysis
6. Image processing (PIL)
7. PDF text extraction
8. API calls

Return a JSON object with:
{
    "steps": [
        {
            "type": "web_scrape|data_process|query|visualize|analyze|custom",
            "description": "What this step does",
            "code": "Python code to execute",
            "dependencies": ["previous_step_results"]
        }
    ],
    "output_format": "array|object|string",
    "expected_outputs": ["description of each expected output"]
}

Make the plan comprehensive and handle edge cases. Generate complete, executable Python code."""

        user_prompt = f"""
Task: {context['questions']}

Available files:
{json.dumps(context['files_info'], indent=2)}

Create a detailed analysis plan to complete this task.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            plan_text = response.choices[0].message.content
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("Could not extract JSON from LLM response")
                
        except Exception as e:
            logger.error(f"Error generating analysis plan: {str(e)}")
            # Fallback simple plan
            return {
                "steps": [
                    {
                        "type": "custom",
                        "description": "Direct LLM analysis",
                        "code": "# Fallback analysis",
                        "dependencies": []
                    }
                ],
                "output_format": "array",
                "expected_outputs": ["Analysis results"]
            }
    
    async def _execute_analysis_plan(self, plan: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute the analysis plan step by step"""
        
        step_results = {}
        global_vars = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'requests': requests,
            'BeautifulSoup': BeautifulSoup,
            'json': json,
            'base64': base64,
            'sqlite3': sqlite3,
            'duckdb': duckdb,
            'Image': Image,
            'context': context,
            'step_results': step_results
        }
        
        # Add dataframes to global vars
        for filename, file_info in context['available_data'].items():
            if 'dataframe' in file_info:
                global_vars[f"df_{filename.replace('.', '_')}"] = file_info['dataframe']
        
        try:
            for i, step in enumerate(plan['steps']):
                logger.info(f"Executing step {i+1}: {step['description']}")
                
                # Execute the code
                local_vars = {}
                exec(step['code'], global_vars, local_vars)
                
                # Store results
                step_results[f"step_{i+1}"] = local_vars.get('result', None)
                
                # Update global vars with local results
                global_vars.update(local_vars)
            
            # Format final output based on plan
            if plan['output_format'] == 'array':
                # Collect all results into an array
                final_results = []
                for key in sorted(step_results.keys()):
                    if step_results[key] is not None:
                        final_results.append(step_results[key])
                return final_results
            elif plan['output_format'] == 'object':
                return step_results
            else:
                return str(step_results)
                
        except Exception as e:
            logger.error(f"Error executing analysis plan: {str(e)}")
            # Fallback to direct LLM analysis
            return await self._fallback_llm_analysis(context)
    
    async def _fallback_llm_analysis(self, context: Dict[str, Any]) -> Any:
        """Fallback analysis using direct LLM processing"""
        
        system_prompt = """You are a data analyst. Analyze the given task and data, then provide the requested output format.

If the task asks for:
- Multiple answers: Return a JSON array
- Single answer: Return the answer directly
- Visualizations: Return "data:image/png;base64,..." format
- Complex analysis: Return a JSON object

Be precise and follow the exact format requested."""

        user_prompt = f"""
Task: {context['questions']}

Available data summary:
{json.dumps([info['summary'] for info in context['files_info']], indent=2)}

Provide the analysis results in the requested format.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            
            # Try to parse as JSON
            try:
                return json.loads(result)
            except:
                return result
                
        except Exception as e:
            logger.error(f"Error in fallback analysis: {str(e)}")
            return ["Error: Could not complete analysis"]

# Initialize agent
agent = DataAnalystAgent()

@app.post("/api/")
async def analyze_data(request: Request):
    """Main API endpoint for data analysis"""
    try:
        # Parse multipart form data
        form = await request.form()
        
        # Extract questions
        questions = None
        files = []
        
        for key, value in form.items():
            if key == "questions.txt" or "question" in key.lower():
                if hasattr(value, 'read'):
                    content = await value.read()
                    questions = content.decode('utf-8')
                else:
                    questions = str(value)
            elif hasattr(value, 'read'):  # It's a file
                files.append(value)
        
        if not questions:
            raise HTTPException(status_code=400, detail="No questions provided")
        
        # Perform analysis
        results = await agent.analyze_task(questions, files)
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the frontend for testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analyst Agent</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            textarea { width: 100%; height: 150px; padding: 10px; border: 1px solid #ddd; }
            input[type="file"] { width: 100%; padding: 10px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 20px; background: #f8f9fa; border: 1px solid #ddd; }
            .loading { display: none; color: #007bff; }
        </style>
    </head>
    <body>
        <h1>Data Analyst Agent</h1>
        <form id="analysisForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="questions">Questions/Task Description:</label>
                <textarea id="questions" name="questions.txt" placeholder="Enter your data analysis questions here..." required></textarea>
            </div>
            
            <div class="form-group">
                <label for="files">Upload Files (optional):</label>
                <input type="file" id="files" name="files" multiple>
                <small>Supports: CSV, JSON, PDF, Images, Text files</small>
            </div>
            
            <button type="submit">Analyze Data</button>
            <div class="loading" id="loading">Analyzing... Please wait (up to 3 minutes)</div>
        </form>
        
        <div id="result" class="result" style="display: none;">
            <h3>Results:</h3>
            <pre id="resultContent"></pre>
        </div>
        
        <script>
            document.getElementById('analysisForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const questions = document.getElementById('questions').value;
                const files = document.getElementById('files').files;
                
                // Add questions as a text file
                formData.append('questions.txt', new Blob([questions], {type: 'text/plain'}), 'questions.txt');
                
                // Add uploaded files
                for (let i = 0; i < files.length; i++) {
                    formData.append(files[i].name, files[i]);
                }
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                
                try {
                    const response = await fetch('/api/', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    document.getElementById('resultContent').textContent = JSON.stringify(result, null, 2);
                    document.getElementById('result').style.display = 'block';
                    
                } catch (error) {
                    document.getElementById('resultContent').textContent = 'Error: ' + error.message;
                    document.getElementById('result').style.display = 'block';
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
