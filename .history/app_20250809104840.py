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
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Data processing
import pandas as pd
import numpy as np

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Web scraping
import requests
from bs4 import BeautifulSoup

# File processing
import PyPDF2
from PIL import Image

# Google Gemini
import google.generativeai as genai
import duckdb

import re
from dataclasses import dataclass

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    gemini_api_key: Optional[str] = os.getenv('GEMINI_API_KEY')
    model_name: str = 'gemini-1.5-flash'
    max_file_size_mb: int = 50
    request_timeout_seconds: int = 180
    debug: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    host: str = os.getenv('HOST', '0.0.0.0')
    port: int = int(os.getenv('PORT', '5000'))


class UniversalDataAnalyst:
    def __init__(self, config: Config):
        self.config = config
        self.gemini_model = self.setup_gemini()
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def setup_gemini(self):
        if not self.config.gemini_api_key:
            logger.warning("GEMINI_API_KEY not found. LLM features disabled.")
            return None
        try:
            genai.configure(api_key=self.config.gemini_api_key)
            return genai.GenerativeModel(self.config.model_name)
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")
            return None

    def detect_dataset_reference(self, text: str) -> Optional[str]:
        dataset_pattern = r"(s3://[^\s]+|[^\s]+?\.(?:parquet|csv|tsv|json|sqlite|db))"
        matches = re.findall(dataset_pattern, text, re.IGNORECASE)
        return matches[0] if matches else None

    def analyze_request(self, questions: str, files: List[Dict[str, Any]]):
        try:
            analysis_context = {}
            dataset_ref = self.detect_dataset_reference(questions)

            # If dataset reference exists, try DuckDB
            if dataset_ref:
                analysis_context = self.build_analysis_context(questions, files, scrape_urls=False)
                duckdb_query = self.generate_duckdb_query(questions, dataset_ref)
                if duckdb_query:
                    df = self.run_duckdb_query(duckdb_query)
                    if df is not None:
                        analysis_context['duckdb_query_result'] = {
                            'type': 'dataframe',
                            'data': df.to_dict('records')
                        }
            else:
                analysis_context = self.build_analysis_context(questions, files, scrape_urls=True)
                if analysis_context.get('has_urls'):
                    analysis_context['scraped_data'] = self.scrape_web_data(analysis_context['urls'])
                if analysis_context.get('has_files'):
                    analysis_context['file_data'] = self.process_files(files)

            if not any(k in analysis_context for k in ['duckdb_query_result', 'file_data', 'scraped_data']):
                analysis_context['internet_search_results'] = self.search_internet(questions)

            return self.generate_analysis_response(questions, analysis_context)

        except Exception as e:
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def search_internet(self, query: str):
        logger.info(f"Internet search placeholder for: {query}")
        return json.dumps([{"title": "Search Placeholder", "content": f"A search was performed for '{query}'"}])

    def generate_duckdb_query(self, user_prompt: str, dataset_ref: str) -> Optional[str]:
        if not self.gemini_model:
            return None
        prompt = f"""
        You are an expert DuckDB SQL generator.
        Dataset location or reference: {dataset_ref}

        Write a single SQL query using DuckDB to answer:
        {user_prompt}

        - Assume the dataset path is directly readable by DuckDB.
        - Include INSTALL/LOAD statements if needed.
        - Return ONLY the SQL query, no extra text.
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            query = response.text.strip().strip("```sql").strip("```").strip()
            return query
        except Exception as e:
            logger.error(f"DuckDB query gen failed: {e}")
            return None

    def run_duckdb_query(self, query: str):
        try:
            con = duckdb.connect(database=':memory:', read_only=False)
            return con.execute(query).fetchdf()
        except Exception as e:
            logger.error(f"DuckDB exec failed: {e}")
            return None

    def build_analysis_context(self, questions: str, files: List[Dict], scrape_urls=True):
        ctx = {'questions': questions, 'has_files': bool(files), 'has_urls': False, 'urls': []}
        if scrape_urls:
            urls = re.findall(r'https?://[^\s<>"\'\]]+', questions)
            if urls:
                ctx['has_urls'] = True
                ctx['urls'] = urls
        return ctx

    def scrape_web_data(self, urls: List[str]):
        scraped_data = {}
        headers = {'User-Agent': 'Mozilla/5.0'}
        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.content, 'html.parser')

                # Extract tables
                tables_data = []
                for table in soup.find_all('table'):
                    try:
                        df = pd.read_html(StringIO(str(table)))[0]
                        df.columns = [re.sub(r'\[.*?\]', '', str(c)).strip() for c in df.columns]
                        tables_data.append(df.to_dict('records'))
                    except:
                        continue

                # Extract lists
                lists_data = []
                for lst in soup.find_all(['ul', 'ol']):
                    items = [li.get_text(strip=True) for li in lst.find_all('li')]
                    if items:
                        lists_data.append(items)

                # Extract headings & paragraphs
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
                scraped_data[url] = {'error': str(e)}
        return scraped_data

    def process_files(self, files: List[Dict[str, Any]]):
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

    def generate_analysis_response(self, questions: str, context: Dict[str, Any]):
        if not self.gemini_model:
            return {"error": "Gemini model not available"}
        prompt = self.build_analysis_prompt(questions, context)
        try:
            resp = self.gemini_model.generate_content(prompt)
            txt = resp.text.strip().removeprefix("```json").removesuffix("```").strip()
            try:
                return json.loads(txt)
            except:
                return {"analysis_text": txt}
        except Exception as e:
            return {"error": str(e)}

    def build_analysis_prompt(self, questions: str, context: Dict[str, Any]):
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


def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    config = Config()
    analyzer = UniversalDataAnalyst(config)
    app.analyzer = analyzer
    app.config_object = config

    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy', 'gemini': bool(app.analyzer.gemini_model)})

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
            return jsonify(app.analyzer.analyze_request(questions, files_to_process))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host=app.config_object.host, port=app.config_object.port, debug=app.config_object.debug)
