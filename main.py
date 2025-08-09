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

class DataAnalystAgent:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.api_key = os.getenv("AIPIPE_TOKEN")
        self.base_url = "https://aipipe.org/openai/v1"

        if not self.api_key or self.api_key == "your-aipipe-token-here":
            self.client = None
            logger.warning("AIPIPE_TOKEN environment variable not set or is a placeholder. API calls will fail.")
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _serialize_output(self, data: Any) -> Any:
        """Recursively converts non-serializable types to JSON-safe types."""
        if isinstance(data, dict):
            return {k: self._serialize_output(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._serialize_output(i) for i in data]
        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient='split')
        if isinstance(data, pd.Series):
            return data.to_list()
        if isinstance(data, np.generic):
            return data.item()
        if isinstance(data, np.ndarray):
            return data.tolist()
        if hasattr(data, 'isoformat'):
            return data.isoformat()
        return data

    async def analyze_task(self, questions: str, files: List[UploadFile] = None) -> Any:
        """Main analysis function that handles any data analysis task"""
        if not self.client:
            raise HTTPException(
                status_code=401,
                detail="AIPIPE_TOKEN environment variable not set or is invalid. The agent cannot function without a valid API key."
            )
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
                 results = await self._fallback_llm_analysis(context)
            else:
                results = await self._execute_analysis_plan(analysis_plan, context)
            
            return self._serialize_output(results)
            
        except HTTPException:
            raise 
        except Exception as e:
            logger.error(f"Error in analyze_task: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
    async def _process_file(self, file: UploadFile) -> Dict[str, Any]:
        """Process uploaded files and extract information"""
        content = await file.read()
        file_info = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "summary": "File content not processed by default"
        }
        
        try:
            if file.content_type == "text/plain":
                file_info["content"] = content.decode('utf-8')
                file_info["summary"] = f"Text file with {len(file_info['content'])} characters."
            elif file.filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                file_info["dataframe"] = df
                file_info["summary"] = f"CSV with {len(df)} rows and {len(df.columns)} columns. Columns: {list(df.columns)}"
            elif file.filename.endswith('.json'):
                file_info["content"] = json.loads(content.decode('utf-8'))
                file_info["summary"] = "JSON data file"
            elif file.content_type.startswith('image/'):
                file_info["content"] = base64.b64encode(content).decode('utf-8')
                file_info["summary"] = f"Image file ({file.content_type})"
            elif file.filename.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                file_info["content"] = text
                file_info["summary"] = f"PDF with {len(pdf_reader.pages)} pages"
        except Exception as e:
            file_info["summary"] = f"Error processing file: {str(e)}"
            logger.error(f"Error processing file {file.filename}: {str(e)}")
        
        return file_info
    
    async def _generate_analysis_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis plan using LLM, ensuring it's robust."""
        system_prompt = """You are a data analyst agent... [same as before] ...""" # Prompt omitted for brevity
        serializable_files_info = [
            {k: v for k, v in info.items() if k not in ['content', 'dataframe']} 
            for info in context.get('files_info', [])
        ]
        user_prompt = f"Task: {context['questions']}\n\nAvailable files summary:\n{json.dumps(serializable_files_info, indent=2)}\n\nCreate a detailed analysis plan..."

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error generating analysis plan: {str(e)}")
            return {}

    async def _execute_analysis_plan(self, plan: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute the analysis plan step by step"""
        # ... function body is the same, but we pass the error to the fallback
        try:
            #...
            return "..."
        except Exception as e:
            logger.error(f"Error executing analysis plan: {str(e)}")
            return await self._fallback_llm_analysis(context, execution_error=e)
    
    async def _fallback_llm_analysis(self, context: Dict[str, Any], execution_error: Optional[Exception] = None) -> Any:
        """Fallback analysis using direct LLM processing if planning/execution fails."""
        logger.warning("Falling back to direct LLM analysis.")
        system_prompt = """You are a data analyst... [same as before] ..."""
        user_prompt = f"Task: {context['questions']}\n\nAvailable data summary:\n{json.dumps([info['summary'] for info in context['files_info']], indent=2)}"
        
        if execution_error:
            user_prompt += f"\n\nThe initial analysis attempt failed with this error: {str(execution_error)}. Please try to answer the original task directly."

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.1
            )
            result = response.choices[0].message.content
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return result
        except Exception as e:
            logger.error(f"Error in fallback analysis: {str(e)}")
            error_message = f"AI model interaction failed: {str(e)}"
            if 'authentication' in str(e).lower():
                error_message = "The AI model returned an authentication error. Please ensure your AIPIPE_TOKEN is correct and valid."
            return {"error": error_message}

# Initialize agent
agent = DataAnalystAgent()

# FastAPI endpoints (/api/, /, /health) remain the same
# ...
@app.post("/api/")
async def analyze_data(request: Request):
    """Main API endpoint for data analysis"""
    try:
        form = await request.form()
        questions = form.get("questions.txt")
        if isinstance(questions, UploadFile):
            questions = (await questions.read()).decode('utf-8')

        if not questions:
            raise HTTPException(status_code=400, detail="No questions provided in 'questions.txt' field.")

        files = [item for key, item in form.items() if isinstance(item, UploadFile) and key != "questions.txt"]
        
        results = await agent.analyze_task(questions, files)
        return JSONResponse(content=results)
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
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
    # HTML content is the same
    return """..."""

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
