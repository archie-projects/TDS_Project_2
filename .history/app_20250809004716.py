#!/usr/bin/env python3
"""
Universal Data Analyst Agent (Synchronous Flask Version)
A generic AI-powered data analysis API that can handle any type of question and file format.
Uses Gemini Flash 1.5 for intelligent analysis, visualization, and web scraping.
"""

import os
import sys
import logging
import traceback
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from io import BytesIO, StringIO

# Web framework
from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
import werkzeug.serving

# Data processing
import pandas as pd
import numpy as np
import openpyxl

# Visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns

# Web scraping
import requests
from bs4 import BeautifulSoup

# File processing
import PyPDF2
from PIL import Image
import docx

# Google Gemini
import google.generativeai as genai

from urllib.parse import urlparse, urljoin
import re
from dataclasses import dataclass

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Configuration ---
@dataclass
class Config:
    """Application configuration"""
    gemini_api_key: Optional[str] = os.getenv('GEMINI_API_KEY')
    model_name: str = 'gemini-1.5-flash'
    max_file_size_mb: int = 50
    request_timeout_seconds: int = 180
    visualization_format: str = "png"
    debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    host: str = os.getenv('HOST', '0.0.0.0')
    port: int = int(os.getenv('PORT', '5000'))


# --- Main Analysis Class ---
class UniversalDataAnalyst:
    """
    Handles all data processing, analysis, and generation logic.
    This is now a standard synchronous class.
    """
    def __init__(self, config: Config):
        self.config = config
        self.gemini_model = self.setup_gemini()

        # Initialize visualization settings
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def setup_gemini(self) -> Optional[genai.GenerativeModel]:
        """Initializes and returns the Gemini AI model."""
        if not self.config.gemini_api_key:
            logger.warning("GEMINI_API_KEY environment variable not found. LLM features will be disabled.")
            return None

        try:
            genai.configure(api_key=self.config.gemini_api_key)
            model = genai.GenerativeModel(self.config.model_name)
            logger.info(f"Gemini model '{self.config.model_name}' initialized successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            return None

    def analyze_request(self, questions: str, files: List[Dict[str, Any]]) -> Any:
        """
        Main synchronous analysis method.
        """
        try:
            analysis_context = self.build_analysis_context(questions, files)

            if analysis_context.get('has_urls'):
                scraped_data = self.scrape_web_data(analysis_context['urls'])
                analysis_context['scraped_data'] = scraped_data

            if analysis_context.get('has_files'):
                processed_data = self.process_files(files)
                analysis_context['file_data'] = processed_data

            result = self.generate_analysis_response(questions, analysis_context)
            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def build_analysis_context(self, questions: str, files: List[Dict]) -> Dict[str, Any]:
        """Builds analysis context from questions and files."""
        context = {
            'questions': questions,
            'has_files': len(files) > 0,
            'requires_visualization': False,
        }
        # Extract URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, questions)
        context['has_urls'] = bool(urls)
        context['urls'] = urls
        
        # Detect visualization intent
        if any(word in questions.lower() for word in ['plot', 'chart', 'graph', 'visualize']):
            context['requires_visualization'] = True
            
        return context

    def scrape_web_data(self, urls: List[str]) -> Dict[str, Any]:
        """Scrapes data from a list of web URLs."""
        scraped_data = {}
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        for url in urls:
            try:
                logger.info(f"Scraping data from: {url}")
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Fix for FutureWarning: wrap literal HTML in StringIO
                tables = [pd.read_html(StringIO(str(table)))[0].to_dict('records') for table in soup.find_all('table')]
                
                scraped_data[url] = {
                    'title': soup.title.string if soup.title else 'No Title',
                    'tables': tables,
                    'text_content': soup.get_text(separator='\n', strip=True)[:5000]
                }
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                scraped_data[url] = {'error': str(e)}
        return scraped_data

    def process_files(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes a list of uploaded files."""
        processed_data = {}
        for file_data in files:
            filename = file_data['filename']
            content = file_data['content']
            ext = Path(filename).suffix.lower()
            logger.info(f"Processing file: {filename} (type: {ext})")
            
            try:
                if ext in ['.csv', '.tsv']:
                    df = pd.read_csv(BytesIO(content), sep=',' if ext == '.csv' else '\t')
                    processed_data[filename] = {'type': 'dataframe', 'data': df.to_dict('records')}
                elif ext in ['.xlsx', '.xls']:
                    xls = pd.ExcelFile(BytesIO(content))
                    processed_data[filename] = {
                        'type': 'excel',
                        'sheets': {name: pd.read_excel(xls, name).to_dict('records') for name in xls.sheet_names}
                    }
                elif ext == '.json':
                    processed_data[filename] = {'type': 'json', 'data': json.loads(content)}
                elif ext == '.pdf':
                    reader = PyPDF2.PdfReader(BytesIO(content))
                    text = "".join(page.extract_text() for page in reader.pages)
                    processed_data[filename] = {'type': 'pdf', 'text': text}
                elif ext in ['.png', '.jpg', '.jpeg']:
                    processed_data[filename] = {
                        'type': 'image',
                        'base64': base64.b64encode(content).decode('utf-8')
                    }
                else: # Default to text
                    processed_data[filename] = {'type': 'text', 'content': content.decode('utf-8', errors='ignore')}
            except Exception as e:
                logger.error(f"Failed to process file {filename}: {e}")
                processed_data[filename] = {'type': 'error', 'error': str(e)}
        return processed_data

    def generate_analysis_response(self, questions: str, context: Dict[str, Any]) -> Any:
        """Generates the main analysis response using the Gemini model."""
        if not self.gemini_model:
            return {"error": "Gemini model is not available. Check API key."}

        prompt = self.build_analysis_prompt(questions, context)
        
        try:
            logger.info("Sending request to Gemini API...")
            response = self.gemini_model.generate_content(prompt)
            logger.info("Received response from Gemini API.")
            
            cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            try:
                result = json.loads(cleaned_text)
            except json.JSONDecodeError:
                return {"analysis_text": cleaned_text}

            # Check if a visualization was requested by the LLM
            if isinstance(result, dict) and "visualization_request" in result:
                viz_request = result.pop("visualization_request")
                image_b64 = self.create_visualization(viz_request, context)
                if image_b64:
                    # Add the image to the final result. The key should match what the user expects.
                    # This part might need adjustment based on the exact user question format.
                    # Assuming the user asks for a field to be populated with the image.
                    for key, value in result.items():
                        if isinstance(value, str) and "base64" in value:
                             result[key] = image_b64
                             break # Stop after finding the first placeholder
                    else: # If no placeholder found, add it
                        result["visualization_result"] = image_b64

            return result

        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return {"error": f"An error occurred with the Gemini API: {e}"}

    def create_visualization(self, viz_request: Dict, context: Dict) -> Optional[str]:
        """Creates a plot based on a request from the LLM and returns a base64 string."""
        logger.info(f"Handling visualization request: {viz_request}")
        
        # Find a suitable dataframe from the context (files or scraped data)
        df = None
        if context.get('file_data'):
            for file_info in context['file_data'].values():
                if file_info['type'] == 'dataframe':
                    df = pd.DataFrame(file_info['data'])
                    break
        if df is None and context.get('scraped_data'):
            for url_data in context['scraped_data'].values():
                if url_data.get('tables'):
                    df = pd.DataFrame(url_data['tables'][0]) # Use the first table
                    break
        
        if df is None:
            logger.warning("No dataframe found for visualization.")
            return None

        try:
            viz_type = viz_request.get("type", "scatter")
            x_col = viz_request.get("x")
            y_col = viz_request.get("y")

            if not x_col or not y_col:
                return None

            # Data cleaning for plotting
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df.dropna(subset=[x_col, y_col], inplace=True)

            fig, ax = plt.subplots(figsize=(10, 6))

            if viz_type == "scatter":
                sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                if viz_request.get("regression", False):
                    sns.regplot(data=df, x=x_col, y=y_col, ax=ax, scatter=False, color='red')
            
            ax.set_title(viz_request.get("title", f"{y_col} vs. {x_col}"))
            plt.tight_layout()

            # Save plot to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            
            # Encode as base64
            image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{image_b64}"

        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            return None


    def build_analysis_prompt(self, questions: str, context: Dict[str, Any]) -> str:
        """Builds the complete prompt string to send to the LLM."""
        prompt_parts = [
            "You are a Universal Data Analyst Agent. Your task is to analyze the provided data and answer the user's questions.",
            "Please provide your response in a single, valid JSON object.",
            f"\n--- USER QUESTIONS ---\n{questions}",
            "\n--- PROVIDED DATA CONTEXT ---"
        ]
        
        if context.get('file_data'):
            prompt_parts.append("\n[UPLOADED FILE DATA]:")
            for filename, data in context['file_data'].items():
                if data['type'] == 'dataframe':
                    df = pd.DataFrame(data['data'])
                    prompt_parts.append(f"- {filename}: A table with columns {df.columns.tolist()} and {len(df)} rows. Here are the first 3 rows: {df.head(3).to_json(orient='records')}")
                else:
                    prompt_parts.append(f"- {filename}: A {data['type']} file.")
        
        if context.get('scraped_data'):
            prompt_parts.append("\n[SCRAPED WEB DATA]:")
            for url, data in context['scraped_data'].items():
                if data.get('tables'):
                    # Provide info and a sample of the first table found
                    table_sample = data['tables'][0][:3] # First 3 rows of the first table
                    prompt_parts.append(f"- From {url}: Found {len(data['tables'])} table(s). Here is a sample of the first table: {json.dumps(table_sample, indent=2)}")
                else:
                    prompt_parts.append(f"- From {url}: No tables found. Text preview: {data.get('text_content', '')[:500]}...")
        
        if not context.get('file_data') and not context.get('scraped_data'):
            prompt_parts.append("\n[DATA]: No files were provided and no web data was scraped. Answer the questions based on general knowledge.")


        prompt_parts.append("\n--- INSTRUCTIONS ---")
        prompt_parts.append("1. Analyze all the provided data in the context of the user's questions.")
        prompt_parts.append("2. Provide a clear, data-driven answer inside a JSON object.")
        prompt_parts.append("3. If a user asks for a plot or visualization, do NOT generate the image data yourself. Instead, include a key named 'visualization_request' in your JSON response. The value should be another JSON object specifying the 'type' (e.g., 'scatter'), 'x' column, 'y' column, and an optional 'title'. Example: \"visualization_request\": {\"type\": \"scatter\", \"x\": \"Rank\", \"y\": \"Peak\", \"title\": \"Film Rank vs. Peak\"}}")
        prompt_parts.append("4. Your final output MUST be a single, valid JSON object.")
        
        return "\n".join(prompt_parts)


# --- Flask Application Factory ---
def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)
    # Make CORS more explicit to allow all origins for all API routes
    CORS(app, resources={r"/api/*": {"origins": "*"}, r"/health": {"origins": "*"}, r"/config": {"origins": "*"}})
    
    # Load config and create analyzer instance
    config = Config()
    analyzer = UniversalDataAnalyst(config)
    
    # Store the single analyzer instance in the app context
    app.analyzer = analyzer
    app.config_object = config

    # --- Routes ---
    @app.route('/')
    def index():
        return send_from_directory('.', 'index.html')

    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'gemini_available': bool(app.analyzer.gemini_model)
        })

    @app.route('/config')
    def get_config():
        """Get system configuration."""
        cfg = app.config_object
        return jsonify({
            'llm_provider': 'gemini',
            'model_name': cfg.model_name,
            'max_file_size_mb': cfg.max_file_size_mb,
            'has_gemini_key': bool(cfg.gemini_api_key),
            'status': 'ready'
        })

    @app.route('/api/', methods=['POST'])
    def analyze_data():
        """Main API endpoint for data analysis."""
        try:
            # The frontend sends all files under the 'files' key.
            # We need to find 'questions.txt' among them.
            all_files = request.files.getlist("files")
            
            if not all_files:
                return jsonify({"error": "No questions provided. Please type a question before analyzing."}), 400

            questions_file = None
            data_files_from_request = []
            
            # Separate the questions file from the data files
            for file_obj in all_files:
                if file_obj.filename == 'questions.txt':
                    questions_file = file_obj
                else:
                    data_files_from_request.append(file_obj)

            if not questions_file:
                return jsonify({"error": "Internal error: questions.txt was not found in the request."}), 400
            
            questions = questions_file.read().decode('utf-8').strip()
            if not questions:
                 return jsonify({"error": "Questions cannot be empty."}), 400

            # Process the actual data files uploaded by the user
            files_to_process = []
            max_size = app.config_object.max_file_size_mb * 1024 * 1024
            for file_obj in data_files_from_request:
                content = file_obj.read()
                if len(content) > max_size:
                     return jsonify({"error": f"File {file_obj.filename} is too large."}), 413
                
                files_to_process.append({
                    'filename': file_obj.filename,
                    'content': content
                })

            logger.info(f"Handling request with {len(files_to_process)} data file(s). Questions: '{questions[:100]}...'")
            
            # Call the synchronous analysis method
            result = app.analyzer.analyze_request(questions, files_to_process)
            
            return jsonify(result)

        except Exception as e:
            logger.error(f"API request failed: {traceback.format_exc()}")
            return jsonify({"error": f"An internal server error occurred: {e}"}), 500

    return app

# --- Main Execution ---
if __name__ == '__main__':
    # Create the Flask app instance
    app = create_app()
    config = app.config_object

    # Check for Gemini API key and provide instructions if missing
    if not config.gemini_api_key:
        print("\n⚠️  WARNING: GEMINI_API_KEY not found!", file=sys.stderr)
        print("Please set your Gemini API key as an environment variable:", file=sys.stderr)
        print("export GEMINI_API_KEY='your-api-key-here'", file=sys.stderr)
        print("The application will run with LLM features disabled.\n", file=sys.stderr)

    print(f"""
    🤖 Universal Data Analyst Agent
    ===============================
    🌐 Server running at: http://{config.host}:{config.port}
    🔧 Debug mode: {config.debug}
    🔍 LLM Model: {config.model_name}
    
    Press Ctrl+C to stop
    """)

    # Use Werkzeug's development server
    app.run(host=config.host, port=config.port, debug=config.debug)
