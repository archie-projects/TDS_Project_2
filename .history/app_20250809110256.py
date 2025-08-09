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

    def detect_dataset_reference(self, text: str) -> Optional[str]:
        """
        Detects a dataset reference in free text.
        Looks for s3:// paths and common dataset file extensions (.parquet, .csv, .json, .sqlite, .db, .tsv).
        Returns the first match or None.
        """
        if not text:
            return None
        pattern = r"(s3://[^\s,;]+|[^\s,;]+?\.(?:parquet|csv|tsv|json|sqlite|db))"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        return matches[0] if matches else None

    def analyze_request(self, questions: str, files: List[Dict[str, Any]]) -> Any:
        """
        Main synchronous analysis method.
        Removed special-case hardcoding for any one dataset and replaced it with generic
        dataset detection. If a dataset reference is found, attempt DuckDB flow.
        """
        try:
            analysis_context = {} # Start with an empty context

            # Detect dataset references generically (S3/path/parquet/csv/json/etc.)
            dataset_ref = self.detect_dataset_reference(questions)

            if dataset_ref:
                logger.info(f"Dataset reference detected: {dataset_ref}. Attempting DuckDB query generation.")
                # Build context without scraping (user explicitly provided dataset)
                analysis_context = self.build_analysis_context(questions, files, scrape_urls=False)

                # Ask LLM to generate a DuckDB query referencing the provided dataset
                duckdb_query = self.generate_duckdb_query(questions, dataset_ref)
                if duckdb_query:
                    query_result_df = self.run_duckdb_query(duckdb_query)
                    if query_result_df is not None:
                        analysis_context['duckdb_query_result'] = {
                            'type': 'dataframe',
                            'data': query_result_df.to_dict('records')
                        }
            else:
                # No explicit dataset reference — proceed with normal scraping / file processing
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

    def generate_duckdb_query(self, user_prompt: str, dataset_ref: Optional[str] = None) -> Optional[str]:
        """Uses Gemini to generate a DuckDB query from a user prompt and an optional dataset reference."""
        if not self.gemini_model:
            return None

        # Provide a general prompt that includes the dataset reference when available.
        # Keep instructions clear: produce a single SQL query only.
        dataset_note = f"Dataset reference: {dataset_ref}\n" if dataset_ref else ""
        prompt = f"""
        You are an expert DuckDB data engineer. Your task is to write a single, efficient SQL query to retrieve all data necessary to answer a user's request.

        {dataset_note}
        The user's request is:
        ---
        {user_prompt}
        ---

        INSTRUCTIONS:
        1. Carefully analyze all the questions the user is asking.
        2. Construct a single DuckDB SQL query that selects all the columns needed to answer every question.
        3. If the dataset reference appears to be a parquet file or S3 parquet, include:
           INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;
           and use read_parquet('<path>') patterns where appropriate.
        4. If the dataset looks like CSV/TSV, assume it can be read using read_csv_auto('<path>').
        5. If date differences are requested, compute them in the SQL using CAST(... AS DATE) - CAST(... AS DATE).
        6. Return ONLY the raw SQL query, without explanation, markdown, or comments.
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
            logger.info(f"Generated DuckDB query: {query[:300]}{'...' if len(query) > 300 else ''}")
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

    from urllib.parse import urlparse

  def clean_url(raw_url: str) -> str:
      # Remove whitespace and control chars
      url = raw_url.strip()
      # Remove trailing punctuation/brackets not part of URL
      url = re.sub(r'[\)\]\}>]+$', '', url)
      return url
    def scrape_web_data(self, urls: List[str]) -> Dict[str, Any]:
        """Scrapes data from a list of web URLs with multiple extraction fallbacks (Wikipedia-friendly + generic)."""
        scraped_data = {}
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        for url in urls:
            try:
                logger.info(f"Scraping data from: {url}")
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')

                tables = []
                parsed_any_table = False

                # First try common helpful table classes (Wikipedia and other structured sites)
                table_classes_to_try = ['wikitable', 'sortable', 'dataframe', 'table', 'wikitable sortable']
                found_tables = []
                for cls in table_classes_to_try:
                    found_tables.extend(soup.find_all('table', {'class': re.compile(cls)}))

                # Add all <table> tags as a fallback (but keep those already found unique)
                all_tables = soup.find_all('table')
                for t in all_tables:
                    if t not in found_tables:
                        found_tables.append(t)

                # Parse each table with pandas.read_html (robust for many table structures)
                for table in found_tables:
                    try:
                        df = pd.read_html(StringIO(str(table)))[0]
                        # Clean up multi-level headers if present
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(-1)
                        df.columns = [re.sub(r'\[.*?\]', '', str(c)).strip() for c in df.columns]
                        tables.append(df.to_dict('records'))
                        parsed_any_table = True
                        logger.info(f"Parsed table with columns: {df.columns.tolist()}")
                    except Exception as e:
                        # If pandas fails for this table, skip but continue trying other tables
                        logger.debug(f"Could not parse a table element with pandas: {e}")

                # If no structured tables parsed, try page-level read_html (some pages expose tables only at page level)
                if not parsed_any_table:
                    try:
                        page_tables = pd.read_html(StringIO(response.text))
                        for df in page_tables:
                            if isinstance(df.columns, pd.MultiIndex):
                                df.columns = df.columns.get_level_values(-1)
                            df.columns = [re.sub(r'\[.*?\]', '', str(c)).strip() for c in df.columns]
                            tables.append(df.to_dict('records'))
                            parsed_any_table = True
                            logger.info("Parsed a table from the page-level read_html fallback.")
                    except Exception:
                        logger.debug("No tables found via page-level read_html fallback.")

                # Extract lists (ul/ol) and definition lists (dl)
                lists = []
                for lst in soup.find_all(['ul', 'ol']):
                    items = [li.get_text(separator=' ', strip=True) for li in lst.find_all('li')]
                    if items:
                        lists.append(items)

                deflists = []
                for dl in soup.find_all('dl'):
                    pairs = []
                    terms = dl.find_all('dt')
                    defs = dl.find_all('dd')
                    for t, d in zip(terms, defs):
                        pairs.append({t.get_text(strip=True): d.get_text(strip=True)})
                    if pairs:
                        deflists.append(pairs)

                # Extract headings and a plain text preview (paragraphs)
                headings = [h.get_text(strip=True) for h in soup.find_all(re.compile('^h[1-6]$'))]
                paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]

                # Trim very long text
                text_preview = "\n".join(paragraphs)[:5000]

                if not tables:
                    logger.info("No parsed tables found on page; returning lists/headings/text fallback.")

                scraped_data[url] = {
                    'title': soup.title.string if soup.title else 'No Title',
                    'tables': tables,
                    'lists': lists,
                    'definition_lists': deflists,
                    'headings': headings,
                    'text_content': text_preview
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
                    # content may be bytes; ensure proper decoding
                    try:
                        json_obj = json.loads(content if isinstance(content, str) else content.decode('utf-8'))
                    except Exception:
                        json_obj = json.loads(content.decode('utf-8', errors='ignore'))
                    processed_data[filename] = {'type': 'json', 'data': json_obj}
                elif ext == '.pdf':
                    reader = PyPDF2.PdfReader(BytesIO(content))
                    text = "".join((page.extract_text() or '') for page in reader.pages)
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
            
            cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            try:
                result = json.loads(cleaned_text)
            except json.JSONDecodeError:
                return {"analysis_text": cleaned_text}

            if isinstance(result, dict) and "visualization_request" in result:
                viz_request = result.pop("visualization_request")
                image_b64 = self.create_visualization(viz_request, context)
                if image_b64:
                    for key, value in result.items():
                        if isinstance(value, str) and "base64" in value.lower():
                             result[key] = image_b64
                             break
                    else:
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
            x_col = viz_request.get("x")
            y_col = viz_request.get("y")

            if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                logger.error(f"Invalid columns for visualization: x='{x_col}', y='{y_col}'")
                return None

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

            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            
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
        prompt_parts.append("3. If a user asks for a plot or visualization, do NOT generate the image data yourself. Instead, include a key named 'visualization_request' in your JSON response. The value should be another JSON object specifying the 'type' (e.g., 'scatter'), 'x' column, 'y' column, and an optional 'title'. The columns MUST exist in the provided data. Example: \"visualization_request\": {\"type\": \"scatter\", \"x\": \"year\", \"y\": \"days_of_delay\", \"title\": \"Year vs. Delay\"}}")
        prompt_parts.append("4. Your final output MUST be a single, valid JSON object.")
        
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
        # Restored: serve index.html from project root as in the original file
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
    app.config['DEBUG'] = True
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
