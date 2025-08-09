import os
import json
import base64
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import tempfile
import io
import re
import warnings

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
import duckdb
from PIL import Image
import PyPDF2
from openai import OpenAI

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

    def _serialize_output(self, data: Any) -> Any:
        """Recursively converts non-serializable types to JSON-safe types."""
        if isinstance(data, dict):
            return {k: self._serialize_output(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._serialize_output(i) for i in data]
        if isinstance(data, pd.DataFrame):
            # Use 'split' orientation for a structured DataFrame representation
            return data.to_dict(orient='split')
        if isinstance(data, pd.Series):
            return data.to_list()
        if isinstance(data, np.generic):
            # Converts numpy types (like int64, float64) to native Python types
            return data.item()
        if isinstance(data, np.ndarray):
            return data.tolist()
        if hasattr(data, 'isoformat'):
            # Handles datetime objects
            return data.isoformat()
        return data

    async def analyze_task(self, questions: str, files: List[UploadFile] = None) -> Any:
        """Main analysis function that handles any data analysis task"""
        try:
            context = {
                "questions": questions,
                "files_info": [],
                "available_data": {}
            }
            
            if files:
                for file in files:
                    file_info = await self._process_file(file)
                    context["files_info"].append(file_info)
                    context["available_data"][file.filename] = file_info
            
            analysis_plan = await self._generate_analysis_plan(context)
            
            if not analysis_plan or not analysis_plan.get("steps"):
                 return await self._fallback_llm_analysis(context)

            results = await self._execute_analysis_plan(analysis_plan, context)
            
            # Ensure final results are JSON serializable before returning
            return self._serialize_output(results)
            
        except Exception as e:
            logger.error(f"Error in analyze_task: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
    async def _process_file(self, file: UploadFile) -> Dict[str, Any]:
        """Process uploaded files and extract information"""
        content = await file.read()
        file_info = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "content": None,
            "summary": "File content not processed by default"
        }
        
        try:
            if file.content_type == "text/plain":
                file_info["content"] = content.decode('utf-8')
                file_info["summary"] = f"Text file with {len(file_info['content'])} characters."
                
            elif file.content_type == "text/csv" or file.filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                file_info["dataframe"] = df
                file_info["summary"] = f"CSV with {len(df)} rows and {len(df.columns)} columns. Columns: {list(df.columns)}"
                
            elif file.content_type == "application/json" or file.filename.endswith('.json'):
                file_info["content"] = json.loads(content.decode('utf-8'))
                file_info["summary"] = "JSON data file"
                
            elif file.content_type.startswith('image/'):
                file_info["content"] = base64.b64encode(content).decode('utf-8')
                file_info["summary"] = f"Image file ({file.content_type})"
                
            elif file.content_type == "application/pdf" or file.filename.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                text = "".join(page.extract_text() for page in pdf_reader.pages)
                file_info["content"] = text
                file_info["summary"] = f"PDF with {len(pdf_reader.pages)} pages"
                
        except Exception as e:
            file_info["summary"] = f"Error processing file: {str(e)}"
            logger.error(f"Error processing file {file.filename}: {str(e)}")
        
        return file_info
    
    async def _generate_analysis_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis plan using LLM, ensuring it's robust."""
        
        system_prompt = """You are a data analyst agent. You create a comprehensive analysis plan based on a task and available data.

Return a JSON object with the following structure:
{
    "steps": [
        {
            "type": "web_scrape|data_process|query|visualize|analyze|custom",
            "description": "A description of what this step does.",
            "code": "Python code to execute for this step. IMPORTANT: The final output of this code MUST be assigned to a variable named 'result'. For example: result = df['age'].mean()",
            "dependencies": []
        }
    ],
    "output_format": "array|object|string",
    "expected_outputs": ["A list describing the final expected outputs."]
}

- Analyze the user's request and the available data summary.
- Create a series of steps to fulfill the request.
- For each step, write the corresponding Python code.
- **Crucially, the code in each step must assign its final output to a variable called `result`.**
- Ensure the overall JSON is valid.
"""
        # Create a serializable version of file info for the prompt, removing large content and non-serializable objects.
        serializable_files_info = [
            {k: v for k, v in info.items() if k not in ['content', 'dataframe']} 
            for info in context.get('files_info', [])
        ]

        user_prompt = f"""
Task: {context['questions']}

Available files summary:
{json.dumps(serializable_files_info, indent=2)}

Create a detailed analysis plan to complete this task.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            plan_text = response.choices[0].message.content
            return json.loads(plan_text)
                
        except Exception as e:
            logger.error(f"Error generating analysis plan: {str(e)}")
            return {} # Return empty dict on failure to trigger fallback

    async def _execute_analysis_plan(self, plan: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute the analysis plan step by step"""
        step_results = {}
        global_vars = {
            'pd': pd, 'np': np, 'plt': plt, 'sns': sns,
            'requests': requests, 'BeautifulSoup': BeautifulSoup,
            'json': json, 'base64': base64, 'duckdb': duckdb,
            'Image': Image, 'context': context, 'step_results': step_results,
            'io': io, 'BytesIO': io.BytesIO
        }
        
        for filename, file_info in context['available_data'].items():
            if 'dataframe' in file_info:
                safe_filename = re.sub(r'[^a-zA-Z0-9_]', '_', filename)
                global_vars[f"df_{safe_filename}"] = file_info['dataframe']
        
        try:
            for i, step in enumerate(plan['steps']):
                logger.info(f"Executing step {i+1}: {step['description']}")
                local_vars = {}
                exec(step['code'], global_vars, local_vars)
                step_results[f"step_{i+1}"] = local_vars.get('result', None)
                global_vars.update(local_vars)
            
            output_format = plan.get('output_format', 'array')
            if output_format == 'array':
                return [res for key, res in sorted(step_results.items()) if res is not None]
            elif output_format == 'object':
                return step_results
            else:
                return str(step_results)
                
        except Exception as e:
            logger.error(f"Error executing analysis plan: {str(e)}")
            return await self._fallback_llm_analysis(context)
    
    async def _fallback_llm_analysis(self, context: Dict[str, Any]) -> Any:
        """Fallback analysis using direct LLM processing if planning/execution fails."""
        logger.warning("Falling back to direct LLM analysis.")
        system_prompt = """You are a data analyst. Analyze the given task and data, then provide the requested output format.
Be precise and follow the exact format requested by the user.
- For multiple answers, use a JSON array.
- For complex reports, use a JSON object.
- For visualizations, describe the plot or return a base64 string if possible.
- For single answers, return the answer directly as a string or number."""

        user_prompt = f"""
Task: {context['questions']}
Available data summary:
{json.dumps([info['summary'] for info in context['files_info']], indent=2)}
Provide the analysis results in the format requested by the original task.
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
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return result
                
        except Exception as e:
            logger.error(f"Error in fallback analysis: {str(e)}")
            return {"error": "Could not complete analysis due to an internal error."}

# Initialize agent
agent = DataAnalystAgent()

@app.post("/api/")
async def analyze_data(request: Request):
    """Main API endpoint for data analysis"""
    try:
        form = await request.form()
        questions = None
        files = []
        
        for key, value in form.items():
            if isinstance(value, str) and "question" in key.lower():
                 questions = value
            elif hasattr(value, 'filename'): # It's an UploadFile
                 if "question" in value.filename:
                     content = await value.read()
                     questions = content.decode('utf-8')
                 else:
                     files.append(value)

        if not questions and "questions.txt" in form:
            questions_file = form["questions.txt"]
            content = await questions_file.read()
            questions = content.decode('utf-8')
        
        if not questions:
            raise HTTPException(status_code=400, detail="No questions provided in 'questions.txt' or a similar field.")
        
        results = await agent.analyze_task(questions, files)
        return JSONResponse(content=results)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"API Error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"An unhandled error occurred: {str(e)}"}
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
            .result { margin-top: 20px; padding: 20px; background: #f8f9fa; border: 1px solid #ddd; white-space: pre-wrap; word-wrap: break-word;}
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
                
                formData.append('questions.txt', new Blob([questions], {type: 'text/plain'}), 'questions.txt');
                
                for (let i = 0; i < files.length; i++) {
                    formData.append(files[i].name, files[i]);
                }
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                
                try {
                    const response = await fetch('/api/', {
                        method: 'POST',
                        body: formData,
                        headers: { 'Accept': 'application/json' },
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        document.getElementById('resultContent').textContent = JSON.stringify(result, null, 2);
                    } else {
                        document.getElementById('resultContent').textContent = 'Error: ' + (result.detail || result.error || JSON.stringify(result));
                    }
                    document.getElementById('result').style.display = 'block';
                    
                } catch (error) {
                    document.getElementById('resultContent').textContent = 'An unexpected error occurred: ' + error.message;
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
