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
import requests
from bs4 import BeautifulSoup
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
from tavily import TavilyClient # Import Tavily

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
import io, base64, matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def encode_plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

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
    model_name: str = 'gemini-2.5-pro'
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
        
        # Initialize Tavily client if API key is present
        if self.config.tavily_api_key:
            self.tavily_client = TavilyClient(api_key=self.config.tavily_api_key)
            logger.info("Tavily client initialized successfully.")
        else:
            self.tavily_client = None
            logger.warning("TAVILY_API_KEY not found. Web search will be disabled.")

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
        """Performs a web search using Tavily and returns a summarized string of results."""
        if not self.tavily_client:
            logger.warning("Tavily client not initialized. Skipping web search.")
            return None
        
        try:
            logger.info(f"Performing Tavily web search for: '{query}'")
            # Using search_depth="advanced" for more comprehensive results
            response = self.tavily_client.search(query=query, search_depth="advanced", max_results=5)
            
            # Format results into a clean string for the LLM to understand
            results_str = "\n\n".join(
                [f"Title: {res['title']}\nURL: {res['url']}\nContent: {res['content']}" for res in response.get('results', [])]
            )
            return results_str
        except Exception as e:
            logger.error(f"Tavily web search failed: {e}")
            return "Web search failed to retrieve results."

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

    def is_visualization_key(self, key: str) -> bool:
        """Detect if a JSON key should contain a visualization."""
        viz_patterns = [
            'graph', 'chart', 'plot', 'histogram', 'heatmap', 'visualization', 
            'network', 'scatter', 'bar', 'line', 'pie', 'box', 'distribution'
        ]
        return any(pattern in key.lower() for pattern in viz_patterns)

    def infer_visualization_from_key_and_data(self, key: str, data_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Infer what type of visualization should be created based on the key name and available data."""
        key_lower = key.lower()
        
        # Get the primary dataframe
        df = self.get_primary_dataframe(data_context)
        if df is None or df.empty:
            return None
            
        # Network/Graph visualizations
        if any(term in key_lower for term in ['network', 'graph']) and not any(term in key_lower for term in ['histogram', 'bar']):
            # Check if data has network structure (source, target columns or similar)
            network_cols = self.detect_network_columns(df)
            if network_cols:
                return {
                    'type': 'network',
                    'source_col': network_cols[0],
                    'target_col': network_cols[1],
                    'title': f'Network Graph'
                }
        
        # Histogram/Distribution
        if any(term in key_lower for term in ['histogram', 'distribution']):
            # Look for degree-related columns or numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'degree' in key_lower and any('degree' in col.lower() for col in df.columns):
                degree_col = next((col for col in df.columns if 'degree' in col.lower()), None)
                if degree_col:
                    return {
                        'type': 'histogram',
                        'column': degree_col,
                        'title': 'Degree Distribution',
                        'color': 'green'
                    }
            elif numeric_cols:
                return {
                    'type': 'histogram',
                    'column': numeric_cols[0],
                    'title': f'Distribution of {numeric_cols[0]}',
                    'color': 'green' if 'degree' in key_lower else 'blue'
                }
        
        # Scatter plot
        if 'scatter' in key_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                return {
                    'type': 'scatter',
                    'x': numeric_cols[0],
                    'y': numeric_cols[1],
                    'title': f'{numeric_cols[0]} vs {numeric_cols[1]}'
                }
        
        # Bar chart
        if 'bar' in key_lower:
            if len(df.columns) >= 2:
                return {
                    'type': 'bar',
                    'x': df.columns[0],
                    'y': df.columns[1],
                    'title': f'{df.columns[1]} by {df.columns[0]}'
                }
        
        return None

    def detect_network_columns(self, df: pd.DataFrame) -> Optional[List[str]]:
        """Detect if dataframe has network/edge structure (source, target columns)."""
        common_network_patterns = [
            ['source', 'target'],
            ['from', 'to'],
            ['node1', 'node2'],
            ['start', 'end'],
            ['src', 'dst']
        ]
        
        cols_lower = [col.lower() for col in df.columns]
        
        for pattern in common_network_patterns:
            if all(p in cols_lower for p in pattern):
                source_col = df.columns[cols_lower.index(pattern[0])]
                target_col = df.columns[cols_lower.index(pattern[1])]
                return [source_col, target_col]
        
        # If we have exactly 2 columns and they seem like they could be edges
        if len(df.columns) == 2:
            return df.columns.tolist()
            
        return None

    def get_primary_dataframe(self, context: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Get the primary dataframe from the analysis context."""
        # Priority: DuckDB result > File data > Scraped data
        if context.get('duckdb_query_result'):
            return pd.DataFrame(context['duckdb_query_result']['data'])
        
        if context.get('file_data'):
            for file_info in context['file_data'].values():
                if file_info['type'] == 'dataframe':
                    return pd.DataFrame(file_info['data'])
        
        if context.get('scraped_data'):
            for url_data in context['scraped_data'].values():
                if url_data.get('tables') and url_data['tables']:
                    return pd.DataFrame(url_data['tables'][0])
        
        return None

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
          
          # Process visualizations - both explicit requests and null values
          result = self.process_and_fill_visualizations(result, context)
          
          return result

      except Exception as e:
          logger.error(f"Gemini analysis failed: {e}")
          return {"error": f"An error occurred with the Gemini API: {e}"}

    def process_and_fill_visualizations(self, result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process visualization requests and fill in null visualization values."""
        
        def process_item(obj, parent_key=None):
            if isinstance(obj, dict):
                processed_dict = {}
                for k, v in obj.items():
                    # Check for explicit visualization requests
                    if isinstance(v, dict) and "type" in v and any(chart_type in str(v.get("type", "")).lower() 
                                                                  for chart_type in ["scatter", "bar", "line", "pie", "histogram", "network"]):
                        processed_dict[k] = self.create_visualization(v, context)
                    
                    # Check for visualization request objects - FIXED: Don't return here, continue processing
                    elif k == "visualization_request" and isinstance(v, dict):
                        processed_dict[k] = self.create_visualization(v, context)
                    
                    # Check for null values in visualization keys
                    elif (v is None or v == "null" or v == "visualization_request") and self.is_visualization_key(k):
                        inferred_viz = self.infer_visualization_from_key_and_data(k, context)
                        if inferred_viz:
                            processed_dict[k] = self.create_visualization(inferred_viz, context)
                        else:
                            processed_dict[k] = None
                    
                    # Handle nested objects
                    else:
                        processed_dict[k] = process_item(v, k)
                
                return processed_dict
            
            elif isinstance(obj, list):
                return [process_item(item, parent_key) for item in obj]
            else:
                return obj
        
        return process_item(result)

    def create_network_graph(self, df: pd.DataFrame, source_col: str, target_col: str, title: str = "Network Graph") -> Optional[str]:
        """Create a network graph visualization."""
        try:
            # Create network graph
            G = nx.from_pandas_edgelist(df, source=source_col, target=target_col)
            
            # Calculate layout
            pos = nx.spring_layout(G, seed=42, k=3, iterations=50)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=1000, alpha=0.7, ax=ax)
            nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')
            
            return self.optimize_image_size(fig)
            
        except Exception as e:
            logger.error(f"Failed to create network graph: {e}")
            return None

    def create_degree_histogram(self, df: pd.DataFrame, title: str = "Degree Distribution") -> Optional[str]:
        """Create a degree histogram from network data."""
        try:
            # If we have network data, calculate degrees
            network_cols = self.detect_network_columns(df)
            if network_cols:
                G = nx.from_pandas_edgelist(df, source=network_cols[0], target=network_cols[1])
                degrees = [d for n, d in G.degree()]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create histogram
                ax.hist(degrees, bins=max(1, len(set(degrees))), 
                       color='green', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Degree')
                ax.set_ylabel('Frequency')
                ax.set_title(title, fontsize=12, fontweight='bold')
                
                return self.optimize_image_size(fig)
            
            # Fallback: if there's a degree column
            elif any('degree' in col.lower() for col in df.columns):
                degree_col = next(col for col in df.columns if 'degree' in col.lower())
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df[degree_col].dropna(), bins=20, color='green', alpha=0.7, edgecolor='black')
                ax.set_xlabel(degree_col)
                ax.set_ylabel('Frequency')
                ax.set_title(title, fontsize=12, fontweight='bold')
                
                return self.optimize_image_size(fig)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create degree histogram: {e}")
            return None

    def create_visualization(self, viz_request: Dict, context: Dict) -> Optional[str]:
        """Generically create any type of plot requested and return base64 PNG under 100kB."""
        if not viz_request:
            return None
            
        logger.info(f"Handling visualization request: {viz_request}")

        # Get a DataFrame from context
        df = self.get_primary_dataframe(context)
        if df is None or df.empty:
            logger.warning("No usable dataframe found for visualization.")
            return None

        try:
            viz_type = viz_request.get("type", "").lower()
            
            # Handle network graphs specially
            if viz_type == "network":
                source_col = viz_request.get("source_col")
                target_col = viz_request.get("target_col")
                title = viz_request.get("title", "Network Graph")
                return self.create_network_graph(df, source_col, target_col, title)
            
            fig, ax = plt.subplots(figsize=(10, 6))

            # Enhanced plotting logic with better error handling
            if viz_type == "histogram":
                column = viz_request.get("column") or viz_request.get("x")
                if column and column in df.columns:
                    color = viz_request.get("color", "blue")
                    ax.hist(df[column].dropna(), bins=20, color=color, alpha=0.7, edgecolor='black')
                    ax.set_xlabel(column)
                    ax.set_ylabel('Frequency')
                else:
                    # Try to create degree histogram if it's a network
                    plt.close(fig)
                    return self.create_degree_histogram(df, viz_request.get("title", "Histogram"))

            elif viz_type == "scatter":
                x_col, y_col = viz_request.get("x"), viz_request.get("y")
                if x_col in df.columns and y_col in df.columns:
                    color = viz_request.get("color", "blue")
                    ax.scatter(df[x_col], df[y_col], color=color, alpha=0.6)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    
                    if viz_request.get("regression"):
                        # Add trend line
                        z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
                        p = np.poly1d(z)
                        ax.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8)
                else:
                    logger.error(f"Scatter plot: Missing columns {x_col}, {y_col}")
                    plt.close(fig)
                    return None

            elif viz_type == "bar":
                x_col, y_col = viz_request.get("x"), viz_request.get("y")
                if x_col in df.columns and y_col in df.columns:
                    color = viz_request.get("bar_color") or viz_request.get("color", "blue")
                    ax.bar(df[x_col], df[y_col], color=color, alpha=0.7)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    # Rotate x-axis labels if they're text and long
                    if df[x_col].dtype == 'object':
                        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                else:
                    logger.error(f"Bar chart: Missing columns {x_col}, {y_col}")
                    plt.close(fig)
                    return None

            elif viz_type == "line":
                x_col, y_col = viz_request.get("x"), viz_request.get("y")
                if x_col in df.columns and y_col in df.columns:
                    color = viz_request.get("line_color") or viz_request.get("color", "blue")
                    ax.plot(df[x_col], df[y_col], marker='o', color=color, linewidth=2)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    
                    # Handle date formatting if x-axis is datetime-like
                    if df[x_col].dtype == 'object':
                        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                else:
                    logger.error(f"Line chart: Missing columns {x_col}, {y_col}")
                    plt.close(fig)
                    return None

            elif viz_type == "pie":
                labels_col, values_col = viz_request.get("labels"), viz_request.get("values")
                if labels_col in df.columns and values_col in df.columns:
                    ax.pie(df[values_col], labels=df[labels_col], autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                else:
                    logger.error(f"Pie chart: Missing columns {labels_col}, {values_col}")
                    plt.close(fig)
                    return None

            elif viz_type == "box" or viz_type == "boxplot":
                column = viz_request.get("column") or viz_request.get("y")
                if column and column in df.columns:
                    ax.boxplot(df[column].dropna())
                    ax.set_ylabel(column)
                else:
                    logger.error(f"Box plot: Column '{column}' not found")
                    plt.close(fig)
                    return None

            else:
                logger.error(f"Unsupported visualization type: {viz_type}")
                plt.close(fig)
                return None

            # Set title
            title = viz_request.get("title", f"{viz_type.title()} Chart")
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Improve layout
            plt.tight_layout()

            return self.optimize_image_size(fig)

        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            logger.error(traceback.format_exc())
            if 'fig' in locals():
                plt.close(fig)
            return None

    def optimize_image_size(self, fig) -> str:
        """Optimize image size to be under 100kB and return base64 string."""
        def encode_png(fig_obj, dpi_val, quality='high'):
            buf = BytesIO()
            if quality == 'high':
                fig_obj.savefig(buf, format='png', dpi=dpi_val, bbox_inches='tight')
            else:
                fig_obj.savefig(buf, format='png', dpi=dpi_val, bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
            buf.seek(0)
            return buf.getvalue()

        # Start with high DPI and reduce until size is acceptable
        dpi = 120
        img_data = encode_png(fig, dpi)
        
        while len(img_data) > 100 * 1024 and dpi > 30:
            dpi = int(dpi * 0.8)
            img_data = encode_png(fig, dpi)
            
        # If still too large, try with lower quality
        if len(img_data) > 100 * 1024:
            dpi = 60
            img_data = encode_png(fig, dpi, quality='low')

        plt.close(fig)
        
        logger.info(f"Generated visualization with size {len(img_data)} bytes")
        return base64.b64encode(img_data).decode('utf-8')

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
            prompt_parts.append(f"- The following information was found by searching the web. Use this to answer the question if no other data is available:")
            prompt_parts.append(context['internet_search_results'])

        if not any(key in context for key in ['file_data', 'scraped_data', 'duckdb_query_result', 'internet_search_results']):
            prompt_parts.append("\n[DATA]: No data was provided or found. Answer the questions based on general knowledge.")

        prompt_parts.append("\n--- INSTRUCTIONS ---")
        prompt_parts.append("1. Analyze all the provided data to answer the user's questions completely and accurately.")
        prompt_parts.append("2. Perform all necessary calculations yourself, such as counting, averaging, finding shortest paths, calculating network metrics, etc.")
        prompt_parts.append("3. For network analysis questions:")
        prompt_parts.append("   - Calculate edge count, node degrees, average degree, density, shortest paths")
        prompt_parts.append("   - Use the actual data structure to determine network properties")
        prompt_parts.append("4. For visualization requests, include the key names exactly as requested by the user (e.g., 'network_graph', 'degree_histogram').")
        prompt_parts.append("5. Set visualization keys to null initially - the system will automatically generate the appropriate visualizations.")
        prompt_parts.append("6. Your final output MUST be a single, valid JSON object with all requested keys.")
        prompt_parts.append("7. Ensure numerical answers are precise and based on the actual data provided.")

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
            # Step 1: Explicitly get the 'questions.txt' file by its required field name.
            # This is the standard way and works perfectly with the curl command.
            questions_file = request.files.get('questions.txt')

            if not questions_file:
                return jsonify({"error": "Mandatory 'questions.txt' file not found in the request. Please ensure it is sent with the correct field name."}), 400
            
            questions = questions_file.read().decode('utf-8').strip()
            if not questions:
                 return jsonify({"error": "The 'questions.txt' file cannot be empty."}), 400

            # Step 2: Process all other data files.
            # We iterate through all items in request.files to find files that are NOT 'questions.txt'.
            files_to_process = []
            max_size = app.config_object.max_file_size_mb * 1024 * 1024
            
            for field_name, file_obj in request.files.items():
                if field_name != 'questions.txt':
                    content = file_obj.read()
                    if len(content) > max_size:
                         return jsonify({"error": f"File {file_obj.filename} is too large."}), 413
                    
                    files_to_process.append({
                        'filename': file_obj.filename,
                        'content': content
                    })

            logger.info(f"Handling request with {len(files_to_process)} data file(s). Questions: '{questions[:100]}...'")
            
            # Step 3: Call the analysis logic.
            result = app.analyzer.analyze_request(questions, files_to_process)
            
            return jsonify(result)

        except Exception as e:
            logger.error(f"API request failed: {traceback.format_exc()}")
            return jsonify({"error": f"An internal server error occurred: {e}"}), 500

    # CORRECTED: This 'return' statement must be the last line INSIDE create_app().
    return app

# CORRECTED: The line below must be at the top level (no indentation).
app = create_app()

# CORRECTED: This block must be at the top level (no indentation).
# --- Main Execution ---
if __name__ == '__main__':
    
    config = app.config_object

    if not config.gemini_api_key:
        print("\nâš ï¸  WARNING: GEMINI_API_KEY not found!", file=sys.stderr)
        print("Please set your Gemini API key as an environment variable:", file=sys.stderr)
        print("export GEMINI_API_KEY='your-api-key-here'", file=sys.stderr)
        print("The application will run with LLM features disabled.\n", file=sys.stderr)
        
    if not config.tavily_api_key:
        print("\nâš ï¸  WARNING: TAVILY_API_KEY not found!", file=sys.stderr)
        print("Web search functionality will be disabled.", file=sys.stderr)

    print(f"""
    🤖 Universal Data Analyst Agent
    ===============================
    🌐 Server running at: http://{config.host}:{config.port}
    🔧 Debug mode: {config.debug}
    📝 LLM Model: {config.model_name}
    
    Press Ctrl+C to stop
    """)
