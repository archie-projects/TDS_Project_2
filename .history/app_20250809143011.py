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

# Web scraping & Internet Search
import requests
from bs4 import BeautifulSoup
# from tavily import TavilyClient # Uncomment to use Tavily for web searches

# File processing
import PyPDF2
from PIL import Image
import docx

# Google Gemini
import google.generativeai as genai
import duckdb

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
    tavily_api_key: Optional[str] = os.getenv('TAVILY_API_KEY') # For general internet search
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
        # if self.config.tavily_api_key:
        #     self.tavily_client = TavilyClient(api_key=self.config.tavily_api_key)
        # else:
        #     self.tavily_client = None

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
            analysis_context = {} # Start with an empty context

            # Special handling for Indian High Court dataset queries
            if "indian high court judgement" in questions.lower() and "s3://indian-high-court-judgments" in questions.lower():
                logger.info("Indian High Court dataset query detected. Generating and executing DuckDB query.")
                analysis_context = self.build_analysis_context(questions, files, scrape_urls=False)
                
                duckdb_query = self.generate_duckdb_query(questions)
                if duckdb_query:
                    query_result_df = self.run_duckdb_query(duckdb_query)
                    if query_result_df is not None:
                        analysis_context['duckdb_query_result'] = {
                            'type': 'dataframe',
                            'data': query_result_df.to_dict('records')
                        }
            else:
                # Original logic for standard file uploads and web scraping
                analysis_context = self.build_analysis_context(questions, files, scrape_urls=True)
                if analysis_context.get('has_urls'):
                    scraped_data = self.scrape_web_data(analysis_context['urls'])
                    analysis_context['scraped_data'] = scraped_data

                if analysis_context.get('has_files'):
                    processed_data = self.process_files(files)
                    analysis_context['file_data'] = processed_data
            
            # If no other data context exists, perform a general internet search
            if not any(key in analysis_context for key in ['duckdb_query_result', 'file_data', 'scraped_data']):
                logger.info("No local data context found. Attempting internet search.")
                search_results = self.search_internet(questions)
                if search_results:
                    analysis_context['internet_search_results'] = search_results

            # Generate the final JSON answer using all available context
            result = self.generate_analysis_response(questions, analysis_context)
            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def search_internet(self, query: str) -> Optional[str]:
        """
        Performs a web search for a given query and returns summarized results.
        NOTE: This is a placeholder for a real search engine API integration.
        """
        logger.info(f"Performing internet search for: {query}")
        # if not self.tavily_client:
        #     logger.warning("Tavily API key not configured. Skipping internet search.")
        #     return None
        try:
            # response = self.tavily_client.search(query=query, search_depth="advanced")
            # return json.dumps(response.get('results', []))
            
            # Placeholder response if Tavily client is not available
            return json.dumps([{"title": "Search Placeholder", "content": f"A web search was performed for '{query}'. The top results would be summarized here to answer the question."}])
        except Exception as e:
            logger.error(f"Internet search failed: {e}")
            return None

    def generate_duckdb_query(self, user_prompt: str) -> Optional[str]:
        """Uses Gemini to generate a DuckDB query from a user prompt."""
        if not self.gemini_model:
            return None
            
        prompt = f"""
        You are an expert DuckDB data engineer. Your task is to write a single, efficient SQL query to retrieve all data necessary to answer a user's request about the Indian High Court Judgments dataset.

        The dataset is available via `read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')`.

        The user's request is:
        ---
        {user_prompt}
        ---

        INSTRUCTIONS:
        1.  Carefully analyze all the questions the user is asking.
        2.  Construct a **single** DuckDB SQL query that selects all the columns needed to answer every question.
        3.  The query's WHERE clause should be broad enough to include all data needed for all questions. For example, if one question asks for years 2019-2022 and another asks for court '33_10', the WHERE clause should be `WHERE year BETWEEN 2019 AND 2022 OR court_code = '33_10'`.
        4.  If the user asks about the time between two dates (e.g., 'date_of_registration' and 'decision_date'), calculate the difference in days directly in the SQL query. Name the calculated column `days_of_delay`. Use `(CAST(decision_date AS DATE) - CAST(date_of_registration AS DATE))` for this calculation.
        5.  The query MUST start with `INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;`
        6.  Return ONLY the raw SQL query, without any explanation, markdown, or comments.
        """
        try:
            logger.info("Generating DuckDB query with Gemini...")
            response = self.gemini_model.generate_content(prompt)
            # Clean up the response to get only the SQL
            query = response.text.strip()
            if query.startswith("```sql"):
                query = query[6:]
            if query.endswith("```"):
                query = query[:-3]
            query = query.strip()
            logger.info(f"Generated DuckDB query: {query}")
            return query
        except Exception as e:
            logger.error(f"Failed to generate DuckDB query: {e}")
            return None

    def run_duckdb_query(self, query: str) -> Optional[pd.DataFrame]:
        """Executes a DuckDB query and returns the result as a DataFrame."""
        try:
            logger.info(f"Executing DuckDB query...")
            con = duckdb.connect(database=':memory:', read_only=False)
            # The query generated by the LLM should contain the INSTALL/LOAD statements.
            result_df = con.execute(query).fetchdf()
            logger.info(f"DuckDB query returned a DataFrame with shape {result_df.shape}")
            return result_df
        except Exception as e:
            logger.error(f"DuckDB query failed: {e}")
            return None

    def build_analysis_context(self, questions: str, files: List[Dict], scrape_urls: bool = True) -> Dict[str, Any]:
        """Builds analysis context from questions and files."""
        context = {
            'questions': questions,
            'has_files': len(files) > 0,
            'requires_visualization': False,
            'has_urls': False,
            'urls': []
        }
        
        if scrape_urls:
            # Extract URLs, excluding any with a trailing parenthesis to avoid errors
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]\)]+'
            urls = re.findall(url_pattern, questions)
            if urls:
                context['has_urls'] = True
                context['urls'] = urls
        
        if any(word in questions.lower() for word in ['plot', 'chart', 'graph', 'visualize']):
            context['requires_visualization'] = True
            
        return context

    def scrape_web_data(self, urls: List[str]) -> Dict[str, Any]:
        """Scrapes data from a list of web URLs, with improved handling for Wikipedia."""
        scraped_data = {}
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        for url in urls:
            try:
                logger.info(f"Scraping data from: {url}")
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the specific, sortable data table on Wikipedia pages
                tables = []
                for table in soup.find_all('table', {'class': 'wikitable'}):
                    try:
                        # Use pandas to parse the table, which is good at handling most structures
                        df = pd.read_html(StringIO(str(table)))[0]
                        
                        # Clean up multi-level column headers if they exist
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(-1) # Get the last level of the multi-index
                        
                        # Clean column names (remove special characters, extra spaces)
                        df.columns = [re.sub(r'\[.*?\]', '', col).strip() for col in df.columns]

                        tables.append(df.to_dict('records'))
                        logger.info(f"Successfully parsed a wikitable with columns: {df.columns.tolist()}")
                    except Exception as e:
                        logger.warning(f"Could not parse a wikitable with pandas, will skip. Error: {e}")

                # If no wikitables found, fall back to generic table search
                if not tables:
                     logger.info("No wikitables found, falling back to generic table search.")
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
                  logger.info("Sending final analysis request to Gemini API...")
                  response = self.gemini_model.generate_content(prompt)
                  logger.info("Received final analysis from Gemini API.")
                  
                  # Clean up the response text more thoroughly
                  cleaned_text = response.text.strip()
                  
                  # Remove various markdown code block markers
                  if cleaned_text.startswith("```json"):
                      cleaned_text = cleaned_text[7:]
                  elif cleaned_text.startswith("```"):
                      cleaned_text = cleaned_text[3:]
                      
                  if cleaned_text.endswith("```"):
                      cleaned_text = cleaned_text[:-3]
                      
                  cleaned_text = cleaned_text.strip()
                  
                  # Log the cleaned text for debugging
                  logger.info(f"Cleaned response text (first 200 chars): {cleaned_text[:200]}...")
                  
                  try:
          # First, check if the response is already a JSON string wrapped in quotes
          if cleaned_text.startswith('"') and cleaned_text.endswith('"'):
              # It's a JSON string, so parse it first to get the actual JSON
              json_string = json.loads(cleaned_text)
              result = json.loads(json_string)
          else:
              result = json.loads(cleaned_text)
          logger.info("Successfully parsed JSON response")
      except json.JSONDecodeError as e:
          logger.error(f"JSON decode error: {e}")
          logger.error(f"Problematic text: {cleaned_text[:500]}...")
          
          # Try to extract JSON from the text if it's embedded
          json_start = cleaned_text.find('{')
          json_end = cleaned_text.rfind('}')
          if json_start != -1 and json_end != -1 and json_end > json_start:
              try:
                  extracted_json = cleaned_text[json_start:json_end+1]
                  # Check if this extracted JSON is also wrapped as a string
                  if extracted_json.startswith('"{') and extracted_json.endswith('}"'):
                      extracted_json = json.loads(extracted_json)
                  result = json.loads(extracted_json)
                  logger.info("Successfully extracted and parsed JSON from response")
              except json.JSONDecodeError:
                  return {"analysis_text": cleaned_text}
          else:
              return {"analysis_text": cleaned_text}

            # Process visualization requests and replace them with actual base64 images
            if isinstance(result, dict):
                # Look for keys that contain visualization_request
                keys_to_process = []
                for key, value in result.items():
                    if isinstance(value, dict) and "visualization_request" in value:
                        keys_to_process.append((key, value["visualization_request"]))
                
                # Process each visualization request
                for key, viz_request in keys_to_process:
                    logger.info(f"Processing visualization request for key '{key}': {viz_request}")
                    image_b64 = self.create_visualization(viz_request, context)
                    if image_b64:
                        result[key] = image_b64
                        logger.info(f"Successfully generated visualization for key '{key}'")
                    else:
                        result[key] = "Failed to generate visualization"
                        logger.warning(f"Failed to generate visualization for key '{key}'")
                
                # Handle legacy single visualization_request
                if "visualization_request" in result:
                    viz_request = result.pop("visualization_request")
                    image_b64 = self.create_visualization(viz_request, context)
                    if image_b64:
                        result["visualization_result"] = image_b64

            return result

        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return {"error": f"An error occurred with the Gemini API: {e}"}

    def create_visualization(self, viz_request: Dict, context: Dict) -> Optional[str]:
        """Creates a plot based on a request from the LLM and returns a base64 string."""
        logger.info(f"Handling visualization request: {viz_request}")
        
        df = None
        # Prioritize using the result from a DuckDB query if it exists
        if context.get('duckdb_query_result'):
            df = pd.DataFrame(context['duckdb_query_result']['data'])
        elif context.get('file_data'):
            for file_info in context['file_data'].values():
                if file_info['type'] == 'dataframe':
                    df = pd.DataFrame(file_info['data'])
                    break
        elif context.get('scraped_data'):
            for url_data in context['scraped_data'].values():
                if url_data.get('tables'):
                    df = pd.DataFrame(url_data['tables'][0])
                    break
        
        if df is None:
            logger.warning("No dataframe found for visualization.")
            return None

        try:
            viz_type = viz_request.get("type", "scatter")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if viz_type == "line":
                x_col = viz_request.get("x")
                y_col = viz_request.get("y")
                line_color = viz_request.get("lineColor", "blue")
                
                if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                    logger.error(f"Invalid columns for line plot: x='{x_col}', y='{y_col}'")
                    return None
                
                # Convert x column to datetime if it looks like a date
                if 'date' in x_col.lower() or any(isinstance(val, str) and '-' in str(val) for val in df[x_col].head()):
                    df[x_col] = pd.to_datetime(df[x_col], errors='coerce')
                
                df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                df.dropna(subset=[x_col, y_col], inplace=True)
                df = df.sort_values(x_col)
                
                ax.plot(df[x_col], df[y_col], color=line_color, linewidth=2)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                
            elif viz_type == "histogram":
                x_col = viz_request.get("x")
                bar_color = viz_request.get("barColor", "blue")
                
                if not x_col or x_col not in df.columns:
                    logger.error(f"Invalid column for histogram: x='{x_col}'")
                    return None
                
                df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
                df.dropna(subset=[x_col], inplace=True)
                
                ax.hist(df[x_col], bins=20, color=bar_color, alpha=0.7, edgecolor='black')
                ax.set_xlabel(x_col)
                ax.set_ylabel('Frequency')
                
            elif viz_type == "scatter":
                x_col = viz_request.get("x")
                y_col = viz_request.get("y")

                if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                    logger.error(f"Invalid columns for visualization: x='{x_col}', y='{y_col}'")
                    return None

                df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
                df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                df.dropna(subset=[x_col, y_col], inplace=True)

                sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                if viz_request.get("regression", False):
                    sns.regplot(data=df, x=x_col, y=y_col, ax=ax, scatter=False, color='red')
            
            ax.set_title(viz_request.get("title", f"Visualization"))
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)

            # Ensure clean base64 encoding
            img_data = buf.getvalue()
            image_b64 = base64.b64encode(img_data).decode('utf-8')
            # Remove any trailing characters that aren't valid base64
            image_b64 = image_b64.rstrip('=').rstrip() + '=' * (4 - len(image_b64.rstrip('=')) % 4) if image_b64.rstrip('=') else image_b64
            return f"data:image/png;base64,{image_b64}"

        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            logger.error(traceback.format_exc())
            return None

    def build_analysis_prompt(self, questions: str, context: Dict[str, Any]) -> str:
        """Builds the complete prompt string to send to the LLM."""
        prompt_parts = [
            "You are a Universal Data Analyst Agent. Your task is to analyze the provided data and answer the user's questions.",
            "You MUST provide your response as a SINGLE, VALID JSON object. Do NOT include any text before or after the JSON.",
            "Do NOT wrap the JSON in markdown code blocks or any other formatting.",
            f"\n--- USER QUESTIONS ---\n{questions}",
            "\n--- PROVIDED DATA CONTEXT ---"
        ]
        
        if context.get('duckdb_query_result'):
            prompt_parts.append("\n[DUCKDB QUERY RESULT DATA]:")
            data = context['duckdb_query_result']
            df = pd.DataFrame(data['data'])
            prompt_parts.append(f"- A query was executed and returned a table with columns {df.columns.tolist()} and {len(df)} rows. Here are the first 3 rows: {df.head(3).to_json(orient='records')}")
        
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
                    table_sample = data['tables'][0][:3]
                    prompt_parts.append(f"- From {url}: Found {len(data['tables'])} table(s). Sample of first table: {json.dumps(table_sample, indent=2)}")
                else:
                    prompt_parts.append(f"- From {url}: No tables found. Text preview: {data.get('text_content', '')[:500]}...")
        
        if context.get('internet_search_results'):
            prompt_parts.append("\n[INTERNET SEARCH RESULTS]:")
            prompt_parts.append(f"- The following information was found by searching the web for your query:")
            prompt_parts.append(context['internet_search_results'])

        if not any(key in context for key in ['file_data', 'scraped_data', 'duckdb_query_result', 'internet_search_results']):
            prompt_parts.append("\n[DATA]: No data was provided or found. Answer the questions based on general knowledge.")


        prompt_parts.append("\n--- INSTRUCTIONS ---")
        prompt_parts.append("1. Analyze all the provided data to answer the user's questions.")
        prompt_parts.append("2. Perform all necessary calculations yourself, such as counting, date differences, and regression analysis, using the provided data.")
        prompt_parts.append("3. If a user asks for a plot or visualization, include a key in your JSON response where the value is another JSON object with 'visualization_request' as a key. The visualization_request should specify the 'type' (e.g., 'line', 'histogram', 'scatter'), 'x' column, 'y' column (if needed), and optional styling like 'lineColor', 'barColor', and 'title'. The columns MUST exist in the provided data.")
        prompt_parts.append("4. For line charts, use type 'line' and specify 'lineColor' if needed.")
        prompt_parts.append("5. For histograms, use type 'histogram' and specify 'barColor' if needed.")
        prompt_parts.append("6. Example visualization request: {\"visualization_request\": {\"type\": \"line\", \"x\": \"date\", \"y\": \"temperature_c\", \"lineColor\": \"red\", \"title\": \"Temperature Over Time\"}}")
        prompt_parts.append("7. CRITICAL: Return ONLY a single, valid JSON object. No explanatory text, no markdown formatting, no code blocks.")
        prompt_parts.append("8. Example response format: {\"average_temp_c\": 25.5, \"temp_line_chart\": {\"visualization_request\": {...}}}")
        
        return "\n".join(prompt_parts)


# --- Flask Application Factory ---
def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}, r"/health": {"origins": "*"}, r"/config": {"origins": "*"}})
    
    config = Config()
    analyzer = UniversalDataAnalyst(config)
    
    app.analyzer = analyzer
    app.config_object = config

    # --- Routes ---
    @app.route('/')
    def index():
        return send_from_directory('.', 'index.html')

    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'gemini_available': bool(app.analyzer.gemini_model)
        })

    @app.route('/config')
    def get_config():
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
            all_files = request.files.getlist("files")
            
            if not all_files:
                return jsonify({"error": "No questions provided. Please type a question before analyzing."}), 400

            questions_file = None
            data_files_from_request = []
            
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
            
            result = app.analyzer.analyze_request(questions, files_to_process)
            
            return jsonify(result)

        except Exception as e:
            logger.error(f"API request failed: {traceback.format_exc()}")
            return jsonify({"error": f"An internal server error occurred: {e}"}), 500

    return app

# --- Main Execution ---
if __name__ == '__main__':
    app = create_app()
    config = app.config_object

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

    app.run(host=config.host, port=config.port, debug=config.debug)
