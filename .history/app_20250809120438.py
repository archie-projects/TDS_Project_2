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
                # Parse each table with pandas.read_html (robust for many table structures)
                largest_table = None
                largest_size = 0

                for table in found_tables:
                    try:
                        df = pd.read_html(StringIO(str(table)))[0]

                        # Clean up multi-level headers if present
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(-1)

                        # Clean bracketed references and strip whitespace
                        clean_cols = [re.sub(r'\[.*?\]', '', str(c)).strip() for c in df.columns]

                        # Deduplicate column names (preserve all instead of dropping duplicates)
                        df.columns = pd.io.parsers.ParserBase({'names': clean_cols})._maybe_dedup_names(clean_cols)

                        # Skip very small or irrelevant tables
                        if df.shape[0] < 5 or df.shape[1] < 3:
                            logger.debug(f"Skipping small/irrelevant table with shape {df.shape}")
                            continue

                        # Track largest table by number of cells
                        size = df.shape[0] * df.shape[1]
                        if size > largest_size:
                            largest_size = size
                            largest_table = df

                        parsed_any_table = True
                        logger.info(f"Parsed table with columns: {list(df.columns)} and {len(df)} rows")

                    except Exception as e:
                        # If pandas fails for this table, skip but continue trying other tables
                        logger.debug(f"Could not parse a table element with pandas: {e}")

                # Append only the largest relevant table
                if largest_table is not None:
                    tables.append(largest_table.to_dict('records'))


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
            elif viz_type == 'pie':
            categories = viz_request.get('categories') or viz_request.get('labels')
            values = viz_request.get('values') or viz_request.get('data')
            if categories and values:
                plt.figure(figsize=(6,6))
                plt.pie(values, labels=categories, autopct='%1.1f%%')
                plt.title(viz_request.get('title', 'Pie Chart'))
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
            else:
                logger.error("Invalid data for pie chart")

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
import os
import re
import json
import base64
import logging
import traceback
from pathlib import Path
from io import BytesIO, StringIO
from urllib.parse import urlparse

import requests
import pandas as pd
import PyPDF2
from PIL import Image
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import duckdb
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ---------------------------
# Config
# ---------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
else:
    model = None
    logger.warning("GEMINI_API_KEY not set. Gemini features disabled.")

# ---------------------------
# Helpers
# ---------------------------
def detect_dataset_reference(text: str):
    dataset_pattern = r"(s3://[^\s]+|[^\s]+?\.(?:parquet|csv|tsv|json|sqlite|db))"
    matches = re.findall(dataset_pattern, text, re.IGNORECASE)
    return matches[0] if matches else None

def build_analysis_context(questions: str, files, scrape_urls=True):
    ctx = {'questions': questions, 'has_files': bool(files), 'has_urls': False, 'urls': []}
    if scrape_urls:
        urls = re.findall(r'https?://[^\s<>"\'\]]+', questions)
        if urls:
            ctx['has_urls'] = True
            ctx['urls'] = urls
    return ctx

def process_files(files):
    processed = {}
    for f in files:
        ext = Path(f['filename']).suffix.lower()
        try:
            if ext in ['.csv', '.tsv']:
                df = pd.read_csv(BytesIO(f['content']), sep=',' if ext == '.csv' else '\t')
                processed[f['filename']] = {'type': 'dataframe', 'data': df.to_dict('records')}
            elif ext in ['.xlsx', '.xls']:
                xls = pd.ExcelFile(BytesIO(f['content']))
                processed[f['filename']] = {
                    'type': 'excel',
                    'sheets': {s: pd.read_excel(xls, s).to_dict('records') for s in xls.sheet_names}
                }
            elif ext == '.json':
                processed[f['filename']] = {'type': 'json', 'data': json.loads(f['content'])}
            elif ext == '.pdf':
                reader = PyPDF2.PdfReader(BytesIO(f['content']))
                text = "".join(page.extract_text() for page in reader.pages)
                processed[f['filename']] = {'type': 'pdf', 'text': text}
            elif ext in ['.png', '.jpg', '.jpeg']:
                processed[f['filename']] = {
                    'type': 'image',
                    'base64': base64.b64encode(f['content']).decode('utf-8')
                }
            else:
                processed[f['filename']] = {'type': 'text', 'content': f['content'].decode('utf-8', errors='ignore')}
        except Exception as e:
            processed[f['filename']] = {'type': 'error', 'error': str(e)}
    return processed

def generate_duckdb_query(user_prompt: str, dataset_ref: str):
    if not model:
        return None
    prompt = f"""
    You are an expert DuckDB SQL generator.
    Dataset location or reference: {dataset_ref}

    Write a single SQL query using DuckDB to answer:
    {user_prompt}

    - Assume the dataset path is directly readable by DuckDB.
    - Return ONLY the SQL query, no extra text.
    """
    try:
        response = model.generate_content(prompt)
        query = response.text.strip().strip("```sql").strip("```").strip()
        return query
    except Exception as e:
        logger.error(f"DuckDB query gen failed: {e}")
        return None

def run_duckdb_query(query: str):
    try:
        con = duckdb.connect(database=':memory:', read_only=False)
        return con.execute(query).fetchdf()
    except Exception as e:
        logger.error(f"DuckDB exec failed: {e}")
        return None

# ---------------------------
# URL Cleaning Helper
# ---------------------------
def clean_url(raw_url: str) -> str:
    """Clean and sanitize a raw URL string."""
    url = raw_url.strip()
    # Remove trailing punctuation/brackets not part of URL
    url = re.sub(r'[\)\]\}>]+$', '', url)
    return url

# ---------------------------
# Web Scraping
# ---------------------------
def scrape_web_data(urls):
    scraped_data = {}
    headers = {'User-Agent': 'Mozilla/5.0'}
    for raw_url in urls:
        url = clean_url(raw_url)
        parsed = urlparse(url)
        if not (parsed.scheme and parsed.netloc):
            scraped_data[raw_url] = {'error': 'Invalid URL format'}
            continue
        try:
            logger.info(f"Scraping data from: {url}")
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, 'html.parser')

            tables_data = []
            for table in soup.find_all('table'):
                try:
                    df = pd.read_html(StringIO(str(table)))[0]
                    df.columns = [re.sub(r'\[.*?\]', '', str(c)).strip() for c in df.columns]
                    tables_data.append(df.to_dict('records'))
                except:
                    continue

            lists_data = []
            for lst in soup.find_all(['ul', 'ol']):
                items = [li.get_text(strip=True) for li in lst.find_all('li')]
                if items:
                    lists_data.append(items)

            headings = [h.get_text(strip=True) for h in soup.find_all(re.compile('^h[1-6]$'))]
            paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]

            scraped_data[url] = {
                'title': soup.title.string if soup.title else '',
                'tables': tables_data,
                'lists': lists_data,
                'headings': headings,
                'text_content': "\n".join(paragraphs)[:5000]
            }
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            scraped_data[url] = {'error': str(e)}
    return scraped_data

# ---------------------------
# Analysis
# ---------------------------
def build_analysis_prompt(questions: str, context):
    parts = [
        "You are a Universal Data Analyst Agent.",
        "Answer in valid JSON only.",
        f"--- USER QUESTIONS ---\n{questions}",
        "--- PROVIDED DATA CONTEXT ---"
    ]
    if context.get('duckdb_query_result'):
        df = pd.DataFrame(context['duckdb_query_result']['data'])
        parts.append(f"DuckDB query returned columns {df.columns.tolist()}, sample: {df.head(3).to_json(orient='records')}")
    if context.get('file_data'):
        for name, data in context['file_data'].items():
            if data['type'] == 'dataframe':
                df = pd.DataFrame(data['data'])
                parts.append(f"File {name}: columns {df.columns.tolist()}, sample: {df.head(3).to_json(orient='records')}")
    if context.get('scraped_data'):
        for url, data in context['scraped_data'].items():
            if data.get('tables'):
                parts.append(f"From {url}: first table sample {json.dumps(data['tables'][0][:3])}")
    if context.get('internet_search_results'):
        parts.append(f"Search results: {context['internet_search_results']}")
    parts.append("--- INSTRUCTIONS ---")
    parts.append("Analyze all provided data to answer the question. Respond ONLY with valid JSON.")
    return "\n".join(parts)

def analyze_request(questions: str, files):
    try:
        analysis_context = {}
        dataset_ref = detect_dataset_reference(questions)

        if dataset_ref:
            analysis_context = build_analysis_context(questions, files, scrape_urls=False)
            duckdb_query = generate_duckdb_query(questions, dataset_ref)
            if duckdb_query:
                df = run_duckdb_query(duckdb_query)
                if df is not None:
                    analysis_context['duckdb_query_result'] = {
                        'type': 'dataframe',
                        'data': df.to_dict('records')
                    }
        else:
            analysis_context = build_analysis_context(questions, files, scrape_urls=True)
            if analysis_context.get('has_urls'):
                analysis_context['scraped_data'] = scrape_web_data(analysis_context['urls'])
            if analysis_context.get('has_files'):
                analysis_context['file_data'] = process_files(files)

        if not any(k in analysis_context for k in ['duckdb_query_result', 'file_data', 'scraped_data']):
            analysis_context['internet_search_results'] = json.dumps(
                [{"title": "Search Placeholder", "content": f"A search was performed for '{questions}'"}]
            )

        if not model:
            return {"error": "Gemini model not available"}

        prompt = build_analysis_prompt(questions, analysis_context)
        resp = model.generate_content(prompt)
        txt = resp.text.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            return json.loads(txt)
        except:
            return {"analysis_text": txt}

    except Exception as e:
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/', methods=['POST'])
def analyze_data():
    try:
        all_files = request.files.getlist("files")
        if not all_files:
            return jsonify({"error": "No files provided"}), 400
        questions_file = None
        data_files = []
        for f in all_files:
            if f.filename == 'questions.txt':
                questions_file = f
            else:
                data_files.append(f)
        if not questions_file:
            return jsonify({"error": "questions.txt missing"}), 400
        questions = questions_file.read().decode('utf-8').strip()
        files_to_process = []
        for f in data_files:
            files_to_process.append({'filename': f.filename, 'content': f.read()})
        return jsonify(analyze_request(questions, files_to_process))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
