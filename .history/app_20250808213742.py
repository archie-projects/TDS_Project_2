#!/usr/bin/env python3
"""
Universal Data Analyst Agent
A generic AI-powered data analysis API that can handle any type of question and file format.
Uses Gemini Flash 1.5 for intelligent analysis, visualization, and web scraping.
"""

import os
import sys
import asyncio
import logging
import traceback
import tempfile
import mimetypes
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from io import BytesIO
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

# Web framework
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import werkzeug.serving

# Data processing
import pandas as pd
import numpy as np
from scipy import stats
import openpyxl  # For Excel files

# Visualization  
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

# Web scraping
import requests
from bs4 import BeautifulSoup
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# File processing
import PyPDF2
from PIL import Image
import docx

# Google Gemini
import google.generativeai as genai

# Database support (for advanced queries)
import sqlite3
import duckdb

# Configuration and utilities
from dataclasses import dataclass
import yaml
import toml
from urllib.parse import urlparse, urljoin
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Application configuration"""
    gemini_api_key: Optional[str] = None
    ollama_model: str = "llama3.1"
    aipipe_token: Optional[str] = None
    max_file_size_mb: int = 50
    request_timeout_seconds: int = 180
    visualization_format: str = "png"
    max_viz_size_kb: int = 100
    chrome_driver_path: Optional[str] = None
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 5000

class UniversalDataAnalyst:
    """
    Universal Data Analyst Agent that can:
    1. Process any file format (CSV, Excel, JSON, PDF, images, etc.)
    2. Scrape web data from any URL
    3. Perform statistical analysis and visualization
    4. Answer any analytical question using LLM
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_gemini()
        self.setup_selenium()
        
        # Initialize visualization settings
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def setup_gemini(self):
        """Initialize Gemini AI"""
        if not self.config.gemini_api_key:
            logger.warning("No Gemini API key provided. LLM features will be limited.")
            self.gemini_model = None
            return
            
        try:
            genai.configure(api_key=self.config.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini 1.5 Flash initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.gemini_model = None
    
    def setup_selenium(self):
        """Setup Selenium for web scraping dynamic content"""
        self.selenium_driver = None
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            
            if self.config.chrome_driver_path:
                self.selenium_driver = webdriver.Chrome(
                    executable_path=self.config.chrome_driver_path, 
                    options=options
                )
            else:
                self.selenium_driver = webdriver.Chrome(options=options)
                
            logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logger.warning(f"Selenium not available: {e}. Web scraping will use requests only.")
    
    async def analyze_request(self, questions: str, files: List[Dict[str, Any]]) -> Any:
        """
        Main analysis method that handles any type of request
        """
        try:
            # Parse questions and determine intent
            analysis_context = await self.build_analysis_context(questions, files)
            
            # Execute analysis based on context
            if analysis_context['has_urls']:
                # Web scraping required
                scraped_data = await self.scrape_web_data(analysis_context['urls'])
                analysis_context['scraped_data'] = scraped_data
            
            if analysis_context['has_files']:
                # File processing required
                processed_data = await self.process_files(files)
                analysis_context['file_data'] = processed_data
            
            # Generate comprehensive answer using Gemini
            result = await self.generate_analysis_response(questions, analysis_context)
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())
            raise

    async def build_analysis_context(self, questions: str, files: List[Dict]) -> Dict[str, Any]:
        """Build analysis context from questions and files"""
        context = {
            'questions': questions,
            'has_files': len(files) > 0,
            'has_urls': False,
            'urls': [],
            'analysis_type': 'general',
            'response_format': 'auto',
            'requires_visualization': False,
            'requires_calculation': False,
            'file_types': []
        }
        
        # Extract URLs from questions
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, questions)
        if urls:
            context['has_urls'] = True
            context['urls'] = urls
        
        # Detect analysis type and requirements
        questions_lower = questions.lower()
        
        if any(word in questions_lower for word in ['plot', 'chart', 'graph', 'visualiz', 'scatter', 'histogram', 'bar chart']):
            context['requires_visualization'] = True
            
        if any(word in questions_lower for word in ['correlat', 'regression', 'mean', 'average', 'count', 'sum', 'statistic']):
            context['requires_calculation'] = True
            
        # Detect response format
        if 'json array' in questions_lower or '[' in questions:
            context['response_format'] = 'json_array'
        elif 'json object' in questions_lower or '{' in questions:
            context['response_format'] = 'json_object'
        elif 'base64' in questions_lower or 'data:image' in questions_lower:
            context['response_format'] = 'mixed'
            context['requires_visualization'] = True
        
        # Analyze file types
        for file_data in files:
            filename = file_data.get('filename', '')
            ext = Path(filename).suffix.lower()
            context['file_types'].append(ext)
        
        return context
    
    async def scrape_web_data(self, urls: List[str]) -> Dict[str, Any]:
        """Scrape data from web URLs"""
        scraped_data = {}
        
        for url in urls:
            try:
                logger.info(f"Scraping data from: {url}")
                
                # Try requests first (faster)
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract tables (most common for data)
                tables = []
                for table in soup.find_all('table'):
                    try:
                        df = pd.read_html(str(table))[0]
                        tables.append(df.to_dict('records'))
                    except:
                        # Manual table parsing
                        rows = []
                        for tr in table.find_all('tr'):
                            row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                            if row:
                                rows.append(row)
                        if rows:
                            tables.append(rows)
                
                # Extract structured data
                scraped_data[url] = {
                    'title': soup.title.get_text(strip=True) if soup.title else '',
                    'tables': tables,
                    'text_content': soup.get_text()[:5000],  # First 5k chars
                    'links': [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)],
                    'images': [urljoin(url, img.get('src')) for img in soup.find_all('img', src=True)]
                }
                
                # Try Selenium for dynamic content if no tables found
                if not tables and self.selenium_driver:
                    try:
                        logger.info(f"Trying Selenium for dynamic content: {url}")
                        self.selenium_driver.get(url)
                        WebDriverWait(self.selenium_driver, 10).until(
                            EC.presence_of_element_located((By.TAG_NAME, "body"))
                        )
                        
                        # Re-parse with Selenium content
                        page_source = self.selenium_driver.page_source
                        soup = BeautifulSoup(page_source, 'html.parser')
                        
                        for table in soup.find_all('table'):
                            try:
                                df = pd.read_html(str(table))[0]
                                scraped_data[url]['tables'].append(df.to_dict('records'))
                            except:
                                pass
                                
                    except Exception as e:
                        logger.warning(f"Selenium scraping failed for {url}: {e}")
                
                logger.info(f"Successfully scraped {len(tables)} tables from {url}")
                
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                scraped_data[url] = {'error': str(e)}
        
        return scraped_data
    
    async def process_files(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process uploaded files of any format"""
        processed_data = {}
        
        for file_data in files:
            filename = file_data['filename']
            content = file_data['content']
            
            try:
                logger.info(f"Processing file: {filename}")
                
                # Determine file type and process accordingly
                ext = Path(filename).suffix.lower()
                
                if ext in ['.csv', '.tsv']:
                    # CSV/TSV files
                    separator = '\t' if ext == '.tsv' else ','
                    df = pd.read_csv(BytesIO(content), sep=separator)
                    processed_data[filename] = {
                        'type': 'dataframe',
                        'data': df.to_dict('records'),
                        'shape': df.shape,
                        'columns': df.columns.tolist(),
                        'dtypes': df.dtypes.to_dict(),
                        'summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else None
                    }
                
                elif ext in ['.xlsx', '.xls']:
                    # Excel files
                    excel_file = pd.ExcelFile(BytesIO(content))
                    sheets_data = {}
                    
                    for sheet_name in excel_file.sheet_names:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        sheets_data[sheet_name] = {
                            'data': df.to_dict('records'),
                            'shape': df.shape,
                            'columns': df.columns.tolist()
                        }
                    
                    processed_data[filename] = {
                        'type': 'excel',
                        'sheets': sheets_data
                    }
                
                elif ext == '.json':
                    # JSON files
                    json_data = json.loads(content.decode('utf-8'))
                    processed_data[filename] = {
                        'type': 'json',
                        'data': json_data
                    }
                
                elif ext == '.pdf':
                    # PDF files
                    pdf_reader = PyPDF2.PdfReader(BytesIO(content))
                    text_content = ""
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() + "\n"
                    
                    processed_data[filename] = {
                        'type': 'pdf',
                        'text': text_content[:10000],  # First 10k chars
                        'num_pages': len(pdf_reader.pages)
                    }
                
                elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                    # Image files
                    image = Image.open(BytesIO(content))
                    processed_data[filename] = {
                        'type': 'image',
                        'size': image.size,
                        'format': image.format,
                        'mode': image.mode,
                        'base64': base64.b64encode(content).decode('utf-8')
                    }
                
                elif ext == '.txt':
                    # Text files
                    text_content = content.decode('utf-8')
                    processed_data[filename] = {
                        'type': 'text',
                        'content': text_content
                    }
                
                else:
                    # Unknown file type - try to read as text
                    try:
                        text_content = content.decode('utf-8')
                        processed_data[filename] = {
                            'type': 'unknown_text',
                            'content': text_content[:5000]
                        }
                    except:
                        processed_data[filename] = {
                            'type': 'binary',
                            'size': len(content),
                            'base64': base64.b64encode(content).decode('utf-8')
                        }
                
                logger.info(f"Successfully processed {filename}")
                
            except Exception as e:
                logger.error(f"Failed to process file {filename}: {e}")
                processed_data[filename] = {
                    'type': 'error',
                    'error': str(e)
                }
        
        return processed_data
    
    async def generate_analysis_response(self, questions: str, context: Dict[str, Any]) -> Any:
        """Generate comprehensive analysis response using Gemini"""
        
        if not self.gemini_model:
            return {"error": "Gemini model not available. Please configure GEMINI_API_KEY."}
        
        try:
            # Build comprehensive prompt
            prompt = self.build_analysis_prompt(questions, context)
            
            # Generate response with Gemini
            logger.info("Generating analysis with Gemini 1.5 Flash...")
            response = await asyncio.to_thread(self.gemini_model.generate_content, prompt)
            
            # Parse and format response
            result = self.parse_gemini_response(response.text, context)
            
            # Add visualizations if required
            if context.get('requires_visualization'):
                result = await self.add_visualizations(result, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            # Fallback to basic analysis
            return await self.fallback_analysis(questions, context)
    
    def build_analysis_prompt(self, questions: str, context: Dict[str, Any]) -> str:
        """Build comprehensive prompt for Gemini"""
        
        prompt_parts = [
            "You are a Universal Data Analyst Agent. Analyze the provided data and answer the questions with precision.",
            f"\nQUESTIONS TO ANSWER:\n{questions}",
            "\nCONTEXT AND DATA:"
        ]
        
        # Add file data context
        if context.get('file_data'):
            prompt_parts.append("\nFILE DATA AVAILABLE:")
            for filename, data in context['file_data'].items():
                if data['type'] == 'dataframe':
                    prompt_parts.append(f"\n- {filename}: DataFrame with shape {data['shape']}")
                    prompt_parts.append(f"  Columns: {data['columns']}")
                    if data['summary']:
                        prompt_parts.append(f"  Summary statistics: {json.dumps(data['summary'], indent=2)}")
                    # Include sample data
                    sample_data = data['data'][:5] if len(data['data']) > 5 else data['data']
                    prompt_parts.append(f"  Sample data: {json.dumps(sample_data, indent=2)}")
                
                elif data['type'] == 'json':
                    prompt_parts.append(f"\n- {filename}: JSON data")
                    prompt_parts.append(f"  Content: {json.dumps(data['data'], indent=2)[:1000]}...")
                
                else:
                    prompt_parts.append(f"\n- {filename}: {data['type']} file")
                    if 'text' in data:
                        prompt_parts.append(f"  Content preview: {data['text'][:500]}...")
        
        # Add scraped data context
        if context.get('scraped_data'):
            prompt_parts.append("\nWEB DATA SCRAPED:")
            for url, data in context['scraped_data'].items():
                if 'tables' in data and data['tables']:
                    prompt_parts.append(f"\n- {url}: Found {len(data['tables'])} tables")
                    for i, table in enumerate(data['tables'][:2]):  # Include first 2 tables
                        prompt_parts.append(f"  Table {i+1}: {json.dumps(table[:5], indent=2)}")
                
                if 'text_content' in data:
                    prompt_parts.append(f"  Text content preview: {data['text_content'][:300]}...")
        
        # Add response format instructions
        response_format = context.get('response_format', 'auto')
        if response_format == 'json_array':
            prompt_parts.append("\nRESPONSE FORMAT: Return your answer as a JSON array: [answer1, answer2, ...]")
        elif response_format == 'json_object':
            prompt_parts.append("\nRESPONSE FORMAT: Return your answer as a JSON object: {\"question1\": \"answer1\", ...}")
        elif response_format == 'mixed':
            prompt_parts.append("\nRESPONSE FORMAT: Return answers in the requested format. For visualizations, include placeholder text like 'VISUALIZATION_REQUIRED_[type]' and I will generate the actual plots.")
        
        # Add analysis instructions
        prompt_parts.append("\nANALYSIS INSTRUCTIONS:")
        prompt_parts.append("1. Provide accurate, data-driven answers")
        prompt_parts.append("2. Perform calculations precisely")
        prompt_parts.append("3. For statistical questions, show your methodology")
        prompt_parts.append("4. For visualizations, describe what should be plotted")
        prompt_parts.append("5. Handle missing or incomplete data gracefully")
        prompt_parts.append("6. Return answers in the exact format requested")
        
        return "\n".join(prompt_parts)
    
    def parse_gemini_response(self, response_text: str, context: Dict[str, Any]) -> Any:
        """Parse Gemini response based on expected format"""
        
        response_text = response_text.strip()
        
        # Try to parse as JSON first
        try:
            # Clean response text
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # If not JSON, return as text
        return response_text
    
    async def add_visualizations(self, result: Any, context: Dict[str, Any]) -> Any:
        """Add actual visualizations to the result"""
        
        try:
            # Look for visualization requirements in the result
            if isinstance(result, list):
                for i, item in enumerate(result):
                    if isinstance(item, str) and ('VISUALIZATION_REQUIRED' in item or 'plot' in item.lower()):
                        viz_data = await self.create_visualization(item, context)
                        if viz_data:
                            result[i] = viz_data
            
            elif isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, str) and ('VISUALIZATION_REQUIRED' in value or 'plot' in value.lower()):
                        viz_data = await self.create_visualization(value, context)
                        if viz_data:
                            result[key] = viz_data
            
            elif isinstance(result, str) and context.get('requires_visualization'):
                # Try to create a basic visualization from available data
                viz_data = await self.create_default_visualization(context)
                if viz_data:
                    if context.get('response_format') == 'json_array':
                        return [result, viz_data]
                    else:
                        return {"analysis": result, "visualization": viz_data}
        
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
        
        return result
    
    async def create_visualization(self, viz_request: str, context: Dict[str, Any]) -> Optional[str]:
        """Create visualization based on request and available data"""
        
        try:
            # Get the first available dataframe
            dataframe_data = None
            for file_data in context.get('file_data', {}).values():
                if file_data.get('type') == 'dataframe' and file_data.get('data'):
                    df = pd.DataFrame(file_data['data'])
                    dataframe_data = df
                    break
            
            # Check scraped data for tables
            if not dataframe_data:
                for scraped_data in context.get('scraped_data', {}).values():
                    if scraped_data.get('tables'):
                        for table in scraped_data['tables']:
                            if isinstance(table, list) and len(table) > 0:
                                df = pd.DataFrame(table)
                                if len(df) > 1:  # Need at least 2 rows
                                    dataframe_data = df
                                    break
                        if dataframe_data is not None:
                            break
            
            if dataframe_data is None:
                logger.warning("No suitable data found for visualization")
                return None
            
            # Create appropriate visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            viz_type = 'scatter'  # Default
            if 'histogram' in viz_request.lower():
                viz_type = 'histogram'
            elif 'bar' in viz_request.lower():
                viz_type = 'bar'
            elif 'scatter' in viz_request.lower() or 'regression' in viz_request.lower():
                viz_type = 'scatter'
            
            # Get numeric columns
            numeric_cols = dataframe_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if viz_type == 'histogram' and len(numeric_cols) > 0:
                dataframe_data[numeric_cols[0]].hist(bins=30, ax=ax)
                ax.set_title(f'Distribution of {numeric_cols[0]}')
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel('Frequency')
                
            elif viz_type == 'bar' and len(dataframe_data.columns) > 1:
                # Use first column for x, second for y
                col1, col2 = dataframe_data.columns[0], dataframe_data.columns[1]
                data_sample = dataframe_data.head(20)  # Limit to 20 bars
                ax.bar(range(len(data_sample)), data_sample.iloc[:, 1])
                ax.set_title(f'{col2} by {col1}')
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                
            elif viz_type == 'scatter' and len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                ax.scatter(dataframe_data[x_col], dataframe_data[y_col], alpha=0.6)
                
                # Add regression line if requested
                if 'regression' in viz_request.lower():
                    z = np.polyfit(dataframe_data[x_col].dropna(), 
                                 dataframe_data[y_col].dropna(), 1)
                    p = np.poly1d(z)
                    ax.plot(dataframe_data[x_col], p(dataframe_data[x_col]), 
                           "r--", alpha=0.8, linewidth=2)
                
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'{y_col} vs {x_col}')
            
            else:
                # Default plot - first numeric column
                if len(numeric_cols) > 0:
                    dataframe_data[numeric_cols[0]].plot(kind='line', ax=ax)
                    ax.set_title(f'Trend of {numeric_cols[0]}')
                else:
                    ax.text(0.5, 0.5, 'No suitable data for visualization', 
                           transform=ax.transAxes, ha='center', va='center')
            
            plt.tight_layout()
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            
            # Check size and compress if needed
            img_data = buffer.getvalue()
            if len(img_data) > self.config.max_viz_size_kb * 1024:
                # Reduce DPI and try again
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=72, bbox_inches='tight')
                buffer.seek(0)
                img_data = buffer.getvalue()
            
            plt.close(fig)
            
            base64_img = base64.b64encode(img_data).decode('utf-8')
            return f"data:image/png;base64,{base64_img}"
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return None
    
    async def create_default_visualization(self, context: Dict[str, Any]) -> Optional[str]:
        """Create a default visualization from available data"""
        return await self.create_visualization("Create a scatter plot", context)
    
    async def fallback_analysis(self, questions: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when Gemini is not available"""
        
        results = {}
        
        # Basic statistical analysis for dataframes
        for filename, file_data in context.get('file_data', {}).items():
            if file_data.get('type') == 'dataframe':
                df = pd.DataFrame(file_data['data'])
                
                results[f"{filename}_summary"] = {
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'columns': df.columns.tolist(),
                }
                
                # Basic statistics for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    results[f"{filename}_stats"] = df[numeric_cols].describe().to_dict()
                    
                    # Correlations
                    if len(numeric_cols) > 1:
                        corr_matrix = df[numeric_cols].corr()
                        results[f"{filename}_correlations"] = corr_matrix.to_dict()
        
        results['analysis_note'] = "This is a fallback analysis. For comprehensive AI-powered analysis, please configure Gemini API key."
        results['questions'] = questions
        
        return results

# Flask Application
def create_app():
    """Create Flask application"""
    
    # Load configuration
    config = Config()
    config.gemini_api_key = os.getenv('GEMINI_API_KEY')
    config.ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.1')
    config.aipipe_token = os.getenv('AIPIPE_TOKEN')
    config.debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    # Initialize analyzer
    analyzer = UniversalDataAnalyst(config)
    
    # Create Flask app
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    # Serve the HTML file at root
    @app.route('/')
    def index():
        """Serve the main HTML interface"""
        # Look for index.html in current directory
        if os.path.exists('index.html'):
            return send_from_directory('.', 'index.html')
        else:
            return jsonify({"error": "index.html not found"}), 404
    
    @app.route('/config')
    def get_config():
        """Get system configuration"""
        return jsonify({
            'llm_provider': 'gemini-1.5-flash',
            'max_file_size_mb': config.max_file_size_mb,
            'request_timeout_seconds': config.request_timeout_seconds,
            'has_gemini_key': bool(config.gemini_api_key),
            'has_aipipe_token': bool(config.aipipe_token),
            'ollama_model': config.ollama_model,
            'visualization_format': config.visualization_format,
            'status': 'ready'
        })
    
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'gemini_available': bool(analyzer.gemini_model),
            'selenium_available': bool(analyzer.selenium_driver)
        })
    
    @app.route('/api/', methods=['POST'])
    def analyze_data():
        """Main API endpoint for data analysis"""
        try:
            # Check if request has files
            if not request.files:
                return jsonify({"error": "No files provided"}), 400
            
            # Extract questions
            questions_file = request.files.get('questions.txt')
            if not questions_file:
                return jsonify({"error": "questions.txt is required"}), 400
            
            questions = questions_file.read().decode('utf-8').strip()
            if not questions:
                return jsonify({"error": "Questions cannot be empty"}), 400
            
            # Process all uploaded files
            files = []
            for file_key, file_obj in request.files.items():
                if file_key == 'questions.txt':
                    continue  # Skip questions file
                
                # Validate file size
                file_obj.seek(0, 2)  # Seek to end
                file_size = file_obj.tell()
                file_obj.seek(0)  # Reset to beginning
                
                if file_size > config.max_file_size_mb * 1024 * 1024:
                    return jsonify({
                        "error": f"File {file_obj.filename} exceeds maximum size of {config.max_file_size_mb}MB"
                    }), 400
                
                files.append({
                    'filename': file_obj.filename,
                    'content': file_obj.read(),
                    'content_type': file_obj.content_type,
                    'size': file_size
                })
            
            logger.info(f"Processing analysis request with {len(files)} files")
            logger.info(f"Questions: {questions[:200]}...")
            
            # Run async analysis in thread pool
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(analyzer.analyze_request(questions, files))
            finally:
                loop.close()
            
            logger.info("Analysis completed successfully")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"API request failed: {e}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    
    @app.errorhandler(413)
    def handle_file_too_large(e):
        return jsonify({"error": "File too large"}), 413
    
    @app.errorhandler(500)
    def handle_internal_error(e):
        return jsonify({"error": "Internal server error"}), 500
    
    # Set maximum content length
    app.config['MAX_CONTENT_LENGTH'] = config.max_file_size_mb * 1024 * 1024 * 10  # 10x for multiple files
    
    return app

def main():
    """Main entry point"""
    
    # Check environment variables
    if not os.getenv('GEMINI_API_KEY'):
        logger.warning("GEMINI_API_KEY not set. Please set it for full functionality.")
        print("\n⚠️  WARNING: GEMINI_API_KEY not found!")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        print("\nYou can get your API key from: https://aistudio.google.com/app/apikey")
        print("The application will still start but with limited functionality.\n")
    
    # Create app
    app = create_app()
    
    # Configure
    config = Config()
    config.debug = os.getenv('DEBUG', 'false').lower() == 'true'
    config.host = os.getenv('HOST', '0.0.0.0')
    config.port = int(os.getenv('PORT', '5000'))
    
    print(f"""
🤖 Universal Data Analyst Agent
===============================
🌐 Server: http://{config.host}:{config.port}
📊 Ready to analyze any data format!
📝 Supports: CSV, Excel, JSON, PDF, Images, Web scraping
🔍 Powered by: Gemini 1.5 Flash
⚡ Max file size: {config.max_file_size_mb}MB
🎯 Timeout: {config.request_timeout_seconds}s

Press Ctrl+C to stop
""")
    
    try:
        # Run server
        if config.debug:
            app.run(host=config.host, port=config.port, debug=True)
        else:
            # Use proper WSGI server for production
            werkzeug.serving.run_simple(
                hostname=config.host,
                port=config.port,
                application=app,
                threaded=True,
                use_reloader=False,
                use_debugger=config.debug
            )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

# Create app instance for WSGI/ASGI servers (e.g., uvicorn, gunicorn)
app = create_app()
