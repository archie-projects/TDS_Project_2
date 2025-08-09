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
              
              tables = []
              
              # Method 1: Try pandas read_html with robust error handling
              try:
                  # Use pandas read_html which is generally more robust
                  dfs = pd.read_html(url, attrs={'class': 'wikitable'})
                  for i, df in enumerate(dfs):
                      try:
                          # Clean up multi-level column headers
                          if isinstance(df.columns, pd.MultiIndex):
                              df.columns = df.columns.get_level_values(-1)
                          
                          # Clean column names (remove special characters, extra spaces, references)
                          df.columns = [re.sub(r'\[.*?\]', '', str(col)).strip() for col in df.columns]
                          
                          # Convert all data to string first to avoid type issues
                          df = df.astype(str)
                          
                          # Clean data: remove references, extra whitespace
                          for col in df.columns:
                              df[col] = df[col].str.replace(r'\[.*?\]', '', regex=True)
                              df[col] = df[col].str.strip()
                              
                          # Try to convert numeric-looking columns back to numbers
                          for col in df.columns:
                              if col.lower() in ['rank', 'peak', 'year'] or 'gross' in col.lower():
                                  # Remove currency symbols and commas
                                  df[col] = df[col].str.replace(r'[\$,]', '', regex=True)
                                  # Try to convert to numeric
                                  df[col] = pd.to_numeric(df[col], errors='ignore')
                          
                          # Only include tables with reasonable amount of data
                          if len(df) > 0 and len(df.columns) > 1:
                              tables.append({
                                  'data': df.to_dict('records'),
                                  'columns': df.columns.tolist(),
                                  'shape': df.shape,
                                  'table_index': i
                              })
                              logger.info(f"Successfully parsed wikitable {i} with columns: {df.columns.tolist()}")
                      
                      except Exception as e:
                          logger.warning(f"Could not process wikitable {i}: {e}")
                          continue
                          
              except Exception as e:
                  logger.warning(f"pandas read_html failed for {url}: {e}")
                  
                  # Method 2: Fallback to BeautifulSoup parsing
                  try:
                      for i, table in enumerate(soup.find_all('table', {'class': 'wikitable'})):
                          rows = []
                          headers = []
                          
                          # Extract headers
                          header_row = table.find('tr')
                          if header_row:
                              headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                              headers = [re.sub(r'\[.*?\]', '', h).strip() for h in headers if h]
                          
                          # Extract data rows
                          for row in table.find_all('tr')[1:]:  # Skip header row
                              cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                              cells = [re.sub(r'\[.*?\]', '', cell).strip() for cell in cells]
                              if len(cells) == len(headers) and any(cells):  # Valid row
                                  rows.append(dict(zip(headers, cells)))
                          
                          if rows and headers:
                              # Create DataFrame and clean it
                              df = pd.DataFrame(rows)
                              
                              # Convert numeric columns
                              for col in df.columns:
                                  if col.lower() in ['rank', 'peak', 'year'] or 'gross' in col.lower():
                                      df[col] = df[col].str.replace(r'[\$,]', '', regex=True)
                                      df[col] = pd.to_numeric(df[col], errors='ignore')
                              
                              tables.append({
                                  'data': df.to_dict('records'),
                                  'columns': df.columns.tolist(),
                                  'shape': df.shape,
                                  'table_index': i
                              })
                              logger.info(f"Successfully parsed wikitable {i} with BeautifulSoup: {df.columns.tolist()}")
                  
                  except Exception as e2:
                      logger.error(f"BeautifulSoup fallback also failed: {e2}")

              scraped_data[url] = {
                  'title': soup.title.string if soup.title else 'No Title',
                  'tables': tables,
                  'text_content': soup.get_text(separator='\n', strip=True)[:5000]
              }
              
              logger.info(f"Successfully scraped {len(tables)} tables from {url}")
              
          except Exception as e:
              logger.error(f"Failed to scrape {url}: {e}")
              scraped_data[url] = {'error': str(e), 'tables': [], 'title': 'Error'}
      
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

          cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
          try:
              result = json.loads(cleaned_text)
          except json.JSONDecodeError:
              return {"analysis_text": cleaned_text}
          


          
          def process_visuals(obj, parent_key=None):
              """Recursively find and replace visualization requests with base64 strings."""
              if isinstance(obj, dict):
                  processed_dict = {}
                  for k, v in obj.items():
                      # Check if this key suggests it's a visualization request
                      is_viz_key = any(viz_word in k.lower() for viz_word in [
                          'chart', 'plot', 'graph', 'histogram', 'scatter', 'bar', 'line', 
                          'pie', 'visualization', 'visual'
                      ])
                      
                      if is_viz_key:
                          if isinstance(v, dict) and "type" in v:
                              # Direct visualization request object
                              processed_dict[k] = self.create_visualization(v, context)
                          elif isinstance(v, dict) and "visualization_request" in v:
                              # Nested visualization request
                              processed_dict[k] = self.create_visualization(v["visualization_request"], context)
                          elif v is None or v == "visualization_request":
                              # Null or placeholder - try to infer from key name
                              viz_request = self.infer_visualization_from_key(k, context)
                              if viz_request:
                                  processed_dict[k] = self.create_visualization(viz_request, context)
                              else:
                                  processed_dict[k] = None
                          else:
                              processed_dict[k] = process_visuals(v, k)
                      else:
                          # Handle nested visualization_request objects
                          if k == "visualization_request" and isinstance(v, dict):
                              return self.create_visualization(v, context)
                          elif isinstance(v, dict) and "type" in v and any(chart_type in str(v.get("type", "")).lower() for chart_type in ["scatter", "bar", "line", "pie", "histogram"]):
                              processed_dict[k] = self.create_visualization(v, context)
                          else:
                              processed_dict[k] = process_visuals(v, k)
                  
                  return processed_dict
              elif isinstance(obj, list):
                  return [process_visuals(i, parent_key) for i in obj]
              else:
                  return obj

          result = process_visuals(result)
          return result

      except Exception as e:
          logger.error(f"Gemini analysis failed: {e}")
          return {"error": f"An error occurred with the Gemini API: {e}"}
      

    def create_visualization(self, viz_request: Dict, context: Dict) -> Optional[str]:
      """Generically create any type of plot requested by the LLM and return base64 PNG under 100kB."""
      if not viz_request:
          return None
          
      logger.info(f"Handling visualization request: {viz_request}")

      # --- Get a DataFrame from context with proper priority ---
      df = None
      data_source = "unknown"
      
      # Priority 1: DuckDB query results
      if context.get('duckdb_query_result'):
          df = pd.DataFrame(context['duckdb_query_result']['data'])
          data_source = "duckdb"
      # Priority 2: Scraped web data (for web scraping requests)
      elif context.get('scraped_data'):
          for url, url_data in context['scraped_data'].items():
              if url_data.get('tables'):
                  for table_info in url_data['tables']:
                      potential_df = pd.DataFrame(table_info['data'])
                      # Check if this table has the required columns
                      required_cols = []
                      if viz_request.get('x'): required_cols.append(viz_request['x'])
                      if viz_request.get('y'): required_cols.append(viz_request['y'])
                      if viz_request.get('column'): required_cols.append(viz_request['column'])
                      if viz_request.get('labels'): required_cols.append(viz_request['labels'])
                      if viz_request.get('values'): required_cols.append(viz_request['values'])
                      
                      if all(col in potential_df.columns for col in required_cols if col):
                          df = potential_df
                          data_source = f"scraped_from_{url}"
                          break
                  if df is not None:
                      break
      # Priority 3: Uploaded file data
      elif context.get('file_data'):
          for filename, file_info in context['file_data'].items():
              if file_info['type'] == 'dataframe':
                  df = pd.DataFrame(file_info['data'])
                  data_source = f"file_{filename}"
                  break

      if df is None or df.empty:
          logger.warning("No usable dataframe found for visualization.")
          return None

      logger.info(f"Using dataframe from {data_source} with columns: {df.columns.tolist()}")

      try:
          viz_type = viz_request.get("type", "").lower()
          
          # Use a larger figure size and better DPI for quality
          fig, ax = plt.subplots(figsize=(12, 8))
          
          # Set style for better appearance
          plt.style.use('default')  # More reliable than seaborn-v0_8
          
          if viz_type == "histogram":
              column = viz_request.get("column") or viz_request.get("x")
              if column and column in df.columns:
                  color = viz_request.get("color", "orange")
                  # Convert to numeric if possible
                  data_to_plot = pd.to_numeric(df[column], errors='coerce').dropna()
                  if len(data_to_plot) > 0:
                      ax.hist(data_to_plot, bins=min(20, len(data_to_plot)//2 + 1), 
                            color=color, alpha=0.7, edgecolor='black', linewidth=1)
                      ax.set_xlabel(column)
                      ax.set_ylabel('Frequency')
                      ax.grid(True, alpha=0.3)
                  else:
                      logger.error(f"No numeric data found in column '{column}'")
                      return None
              else:
                  logger.error(f"Histogram: Column '{column}' not found in dataframe")
                  return None

          elif viz_type == "scatter":
              x_col, y_col = viz_request.get("x"), viz_request.get("y")
              if x_col in df.columns and y_col in df.columns:
                  # Convert to numeric
                  x_data = pd.to_numeric(df[x_col], errors='coerce')
                  y_data = pd.to_numeric(df[y_col], errors='coerce')
                  
                  # Remove NaN values
                  valid_mask = ~(x_data.isna() | y_data.isna())
                  x_data = x_data[valid_mask]
                  y_data = y_data[valid_mask]
                  
                  if len(x_data) > 0:
                      color = viz_request.get("color", "blue")
                      ax.scatter(x_data, y_data, color=color, alpha=0.6, s=50)
                      ax.set_xlabel(x_col)
                      ax.set_ylabel(y_col)
                      ax.grid(True, alpha=0.3)
                      
                      if viz_request.get("regression") or "regression" in viz_request.get("title", "").lower():
                          # Add trend line
                          if len(x_data) > 1:
                              z = np.polyfit(x_data, y_data, 1)
                              p = np.poly1d(z)
                              ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2, label='Trend')
                              ax.legend()
                  else:
                      logger.error(f"No valid numeric data for scatter plot")
                      return None
              else:
                  logger.error(f"Scatter plot: Missing columns {x_col}, {y_col} in {df.columns.tolist()}")
                  return None

          elif viz_type == "bar":
              x_col, y_col = viz_request.get("x"), viz_request.get("y")
              if x_col in df.columns and y_col in df.columns:
                  color = viz_request.get("bar_color") or viz_request.get("color", "blue")
                  
                  # Limit to top N entries if too many
                  if len(df) > 20:
                      df_plot = df.nlargest(20, y_col) if pd.api.types.is_numeric_dtype(df[y_col]) else df.head(20)
                  else:
                      df_plot = df
                  
                  ax.bar(df_plot[x_col], pd.to_numeric(df_plot[y_col], errors='coerce'), 
                        color=color, alpha=0.7)
                  ax.set_xlabel(x_col)
                  ax.set_ylabel(y_col)
                  ax.grid(True, alpha=0.3, axis='y')
                  
                  # Rotate x-axis labels if they're text and long
                  if df[x_col].dtype == 'object':
                      plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
              else:
                  logger.error(f"Bar chart: Missing columns {x_col}, {y_col}")
                  return None

          elif viz_type == "line":
              x_col, y_col = viz_request.get("x"), viz_request.get("y")
              if x_col in df.columns and y_col in df.columns:
                  color = viz_request.get("line_color") or viz_request.get("color", "blue")
                  
                  # Convert to numeric and sort by x if possible
                  x_data = pd.to_numeric(df[x_col], errors='coerce')
                  y_data = pd.to_numeric(df[y_col], errors='coerce')
                  
                  valid_mask = ~(x_data.isna() | y_data.isna())
                  if valid_mask.sum() > 0:
                      x_clean = x_data[valid_mask]
                      y_clean = y_data[valid_mask]
                      
                      # Sort by x-axis for proper line plotting
                      sort_idx = x_clean.argsort()
                      x_sorted = x_clean.iloc[sort_idx]
                      y_sorted = y_clean.iloc[sort_idx]
                      
                      ax.plot(x_sorted, y_sorted, marker='o', color=color, linewidth=2, markersize=4)
                      ax.set_xlabel(x_col)
                      ax.set_ylabel(y_col)
                      ax.grid(True, alpha=0.3)
                  else:
                      logger.error(f"No valid numeric data for line chart")
                      return None
              else:
                  logger.error(f"Line chart: Missing columns {x_col}, {y_col}")
                  return None

          elif viz_type == "pie":
              labels_col, values_col = viz_request.get("labels"), viz_request.get("values")
              if labels_col in df.columns and values_col in df.columns:
                  # Convert values to numeric
                  values = pd.to_numeric(df[values_col], errors='coerce').dropna()
                  labels = df[labels_col][values.index]
                  
                  if len(values) > 0:
                      # Limit to top 10 for readability
                      if len(values) > 10:
                          top_idx = values.nlargest(10).index
                          values = values[top_idx]
                          labels = labels[top_idx]
                      
                      ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                      ax.axis('equal')
                  else:
                      logger.error(f"No valid numeric data for pie chart")
                      return None
              else:
                  logger.error(f"Pie chart: Missing columns {labels_col}, {values_col}")
                  return None

          elif viz_type == "box" or viz_type == "boxplot":
              column = viz_request.get("column") or viz_request.get("y")
              if column and column in df.columns:
                  data_to_plot = pd.to_numeric(df[column], errors='coerce').dropna()
                  if len(data_to_plot) > 0:
                      ax.boxplot(data_to_plot)
                      ax.set_ylabel(column)
                      ax.grid(True, alpha=0.3)
                  else:
                      logger.error(f"No numeric data for box plot")
                      return None
              else:
                  logger.error(f"Box plot: Column '{column}' not found")
                  return None

          else:
              logger.error(f"Unsupported visualization type: {viz_type}")
              return None

          # Set title with better formatting
          title = viz_request.get("title", f"{viz_type.title()} Chart")
          ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
          
          # Improve layout
          plt.tight_layout()

          # --- Save PNG and optimize size ---
          def encode_png(fig_obj, dpi_val, optimize=False):
              buf = BytesIO()
              fig_obj.savefig(buf, format='png', dpi=dpi_val, bbox_inches='tight',
                            facecolor='white', edgecolor='none',
                            optimize=optimize if optimize else False)
              buf.seek(0)
              return buf.getvalue()

          # Start with reasonable DPI and reduce if needed
          dpi = 100
          img_data = encode_png(fig, dpi)
          
          # Reduce DPI until size is under 100KB
          max_size = 100 * 1024
          while len(img_data) > max_size and dpi > 40:
              dpi = int(dpi * 0.85)
              img_data = encode_png(fig, dpi)
              
          # Final attempt with optimization if still too large
          if len(img_data) > max_size and dpi > 30:
              dpi = 50
              img_data = encode_png(fig, dpi, optimize=True)

          plt.close(fig)
          
          logger.info(f"Generated {viz_type} visualization with size {len(img_data)} bytes at {dpi} DPI")
          return base64.b64encode(img_data).decode('utf-8')

      except Exception as e:
          logger.error(f"Failed to create visualization: {e}")
          logger.error(traceback.format_exc())
          if 'fig' in locals():
              plt.close(fig)
          return None


    def infer_visualization_from_key(self, key_name: str, context: Dict[str, Any]) -> Optional[Dict]:
      """Infer visualization parameters from the key name when LLM returns null."""
      key_lower = key_name.lower()
      
      # Get available DataFrame
      df = None
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
      
      if df is None or df.empty:
          return None
      
      columns = df.columns.tolist()
      
      # Infer visualization type and parameters based on key name
      if 'histogram' in key_lower:
          # Find column that might be for histogram
          target_col = None
          if 'temp' in key_lower:
              target_col = next((col for col in columns if 'temp' in col.lower()), None)
          elif 'precip' in key_lower:
              target_col = next((col for col in columns if 'precip' in col.lower() or 'rain' in col.lower()), None)
          
          if target_col:
              return {
                  "type": "histogram",
                  "column": target_col,
                  "title": f"Histogram of {target_col}",
                  "color": "orange" if 'precip' in key_lower else "blue"
              }
      
      elif 'line' in key_lower and 'chart' in key_lower:
          # Find time/date column and value column
          time_col = next((col for col in columns if any(time_word in col.lower() for time_word in ['date', 'time', 'year', 'month'])), None)
          value_col = None
          
          if 'temp' in key_lower:
              value_col = next((col for col in columns if 'temp' in col.lower()), None)
          elif 'precip' in key_lower:
              value_col = next((col for col in columns if 'precip' in col.lower() or 'rain' in col.lower()), None)
          
          if time_col and value_col:
              return {
                  "type": "line",
                  "x": time_col,
                  "y": value_col,
                  "title": f"{value_col} over {time_col}",
                  "line_color": "red" if 'temp' in key_lower else "blue"
              }
      
      elif 'scatter' in key_lower:
          # Try to find two numeric columns
          numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
          if len(numeric_cols) >= 2:
              return {
                  "type": "scatter",
                  "x": numeric_cols[0],
                  "y": numeric_cols[1],
                  "title": f"Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}"
              }
      
      elif 'bar' in key_lower:
          # Find categorical and numeric columns
          categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
          numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
          
          if categorical_cols and numeric_cols:
              return {
                  "type": "bar",
                  "x": categorical_cols[0],
                  "y": numeric_cols[0],
                  "title": f"Bar chart: {categorical_cols[0]} vs {numeric_cols[0]}"
              }
      
      return None
    def build_analysis_prompt(self, questions: str, context: Dict[str, Any]) -> str:
    """Builds the complete prompt string to send to the LLM."""
    prompt_parts = [
        "You are a Universal Data Analyst Agent. Your task is to analyze the provided data and answer the user's questions.",
        "Please provide your response as a single, valid JSON object.",
        f"\n--- USER QUESTIONS ---\n{questions}",
        "\n--- PROVIDED DATA CONTEXT ---"
    ]
    
    available_tables = []
    
    if context.get('duckdb_query_result'):
        prompt_parts.append("\n[DUCKDB QUERY RESULT DATA]:")
        data = context['duckdb_query_result']
        df = pd.DataFrame(data['data'])
        prompt_parts.append(f"- Query result: {len(df)} rows with columns {df.columns.tolist()}")
        prompt_parts.append(f"- Sample data: {df.head(3).to_json(orient='records')}")
        available_tables.append(("duckdb_result", df.columns.tolist()))
    
    if context.get('scraped_data'):
        prompt_parts.append("\n[SCRAPED WEB DATA]:")
        for url, data in context['scraped_data'].items():
            if data.get('tables'):
                prompt_parts.append(f"- From {url}:")
                for i, table_info in enumerate(data['tables']):
                    columns = table_info['columns']
                    shape = table_info['shape']
                    sample_data = table_info['data'][:3]
                    prompt_parts.append(f"  Table {i+1}: {shape[0]} rows × {shape[1]} cols")
                    prompt_parts.append(f"  Columns: {columns}")
                    prompt_parts.append(f"  Sample: {json.dumps(sample_data, indent=4)}")
                    available_tables.append((f"scraped_table_{i+1}", columns))
            else:
                prompt_parts.append(f"- From {url}: No tables found")
    
    if context.get('file_data'):
        prompt_parts.append("\n[UPLOADED FILE DATA]:")
        for filename, data in context['file_data'].items():
            if data['type'] == 'dataframe':
                df = pd.DataFrame(data['data'])
                prompt_parts.append(f"- {filename}: {len(df)} rows with columns {df.columns.tolist()}")
                prompt_parts.append(f"- Sample data: {df.head(3).to_json(orient='records')}")
                available_tables.append((filename, df.columns.tolist()))
            else:
                prompt_parts.append(f"- {filename}: {data['type']} file")
    
    if context.get('internet_search_results'):
        prompt_parts.append("\n[INTERNET SEARCH RESULTS]:")
        prompt_parts.append(context['internet_search_results'])

    if not any(key in context for key in ['file_data', 'scraped_data', 'duckdb_query_result', 'internet_search_results']):
        prompt_parts.append("\n[DATA]: No data was provided or found. Answer based on general knowledge.")

    prompt_parts.append("\n--- AVAILABLE DATA SUMMARY ---")
    if available_tables:
        for table_name, columns in available_tables:
            prompt_parts.append(f"- {table_name}: {columns}")
    else:
        prompt_parts.append("- No tabular data available")

    prompt_parts.append("\n--- ANALYSIS INSTRUCTIONS ---")
    prompt_parts.append("1. Analyze all provided data to answer the user's questions completely.")
    prompt_parts.append("2. Perform calculations yourself using the provided data.")
    prompt_parts.append("3. For visualization requests, use these exact formats:")
    prompt_parts.append("   - Histogram: {\"type\": \"histogram\", \"column\": \"column_name\", \"color\": \"orange\", \"title\": \"Title\"}")
    prompt_parts.append("   - Line chart: {\"type\": \"line\", \"x\": \"x_column\", \"y\": \"y_column\", \"line_color\": \"red\", \"title\": \"Title\"}")
    prompt_parts.append("   - Scatter plot: {\"type\": \"scatter\", \"x\": \"x_column\", \"y\": \"y_column\", \"color\": \"blue\", \"title\": \"Title\", \"regression\": true}")
    prompt_parts.append("   - Bar chart: {\"type\": \"bar\", \"x\": \"category_column\", \"y\": \"value_column\", \"color\": \"green\", \"title\": \"Title\"}")
    prompt_parts.append("   - Pie chart: {\"type\": \"pie\", \"labels\": \"label_column\", \"values\": \"value_column\", \"title\": \"Title\"}")
    prompt_parts.append("4. CRITICAL: Use ONLY column names that exist in the available data shown above.")
    prompt_parts.append("5. If multiple visualizations are needed, include each as a separate key with its own visualization_request object.")
    prompt_parts.append("6. Return a single valid JSON object containing all answers and visualization requests.")
    
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
