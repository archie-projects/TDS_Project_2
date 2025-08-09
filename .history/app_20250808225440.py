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
from io import BytesIO

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
                
                # Simple text and table extraction
                tables = [pd.read_html(str(table))[0].to_dict('records') for table in soup.find_all('table')]
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
            # This is now a synchronous, blocking call
            response = self.gemini_model.generate_content(prompt)
            logger.info("Received response from Gemini API.")
            
            # Attempt to parse the response as JSON, otherwise return as text
            cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                return {"analysis_text": cleaned_text}

        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return {"error": f"An error occurred with the Gemini API: {e}"}

    def build_analysis_prompt(self, questions: str, context: Dict[str, Any]) -> str:
        """Builds the complete prompt string to send to the LLM."""
        prompt_parts = [
            "You are a Universal Data Analyst Agent. Your task is to analyze the provided data and answer the user's questions.",
            "Please provide your response in JSON format.",
            f"\n--- USER QUESTIONS ---\n{questions}",
            "\n--- PROVIDED DATA CONTEXT ---"
        ]
        
        if context.get('file_data'):
            prompt_parts.append("\n[FILE DATA]:")
            # Provide a summary of file data instead of the full content to save tokens
            for filename, data in context['file_data'].items():
                if data['type'] == 'dataframe':
                    df = pd.DataFrame(data['data'])
                    prompt_parts.append(f"- {filename}: A table with columns {df.columns.tolist()} and {len(df)} rows. Here are the first 3 rows: {df.head(3).to_json(orient='records')}")
                elif data['type'] == 'text':
                     prompt_parts.append(f"- {filename}: A text file. Preview: {data['content'][:500]}...")
                else:
                    prompt_parts.append(f"- {filename}: A {data['type']} file.")

        if context.get('scraped_data'):
            prompt_parts.append("\n[WEB DATA]:")
            for url, data in context['scraped_data'].items():
                prompt_parts.append(f"- {url}: Web page with title '{data.get('title', '')}'. Text preview: {data.get('text_content', '')[:500]}...")

        prompt_parts.append("\n--- INSTRUCTIONS ---")
        prompt_parts.append("1. Analyze all the provided data in the context of the user's questions.")
        prompt_parts.append("2. Provide a clear, data-driven answer.")
        prompt_parts.append("3. Your final output MUST be a single, valid JSON object.")
        
        return "\n".join(prompt_parts)


# --- Flask Application Factory ---
def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)
    CORS(app)
    
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
            questions_file = request.files.get('questions.txt')
            if not questions_file:
                return jsonify({"error": "questions.txt is required"}), 400
            questions = questions_file.read().decode('utf-8').strip()

            files = []
            max_size = app.config_object.max_file_size_mb * 1024 * 1024
            for file_key, file_obj in request.files.items():
                if file_key == 'questions.txt':
                    continue
                
                content = file_obj.read()
                if len(content) > max_size:
                     return jsonify({"error": f"File {file_obj.filename} is too large."}), 413
                
                files.append({
                    'filename': file_obj.filename,
                    'content': content
                })

            logger.info(f"Handling request with {len(files)} files. Questions: '{questions[:100]}...'")
            
            # Call the synchronous analysis method
            result = app.analyzer.analyze_request(questions, files)
            
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
