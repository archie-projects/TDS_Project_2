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
import io as _io

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

# Graphs
import networkx as nx

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
    tavily_api_key: Optional[str] = os.getenv('TAVILY_API_KEY')  # For general internet search
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

        # Initialize Tavily client if API key is present (optional)
        try:
            from tavily import TavilyClient  # local import to avoid failing when package missing
            if self.config.tavily_api_key:
                self.tavily_client = TavilyClient(api_key=self.config.tavily_api_key)
                logger.info("Tavily client initialized successfully.")
            else:
                self.tavily_client = None
                logger.warning("TAVILY_API_KEY not found. Web search will be disabled.")
        except Exception:
            self.tavily_client = None
            logger.warning("tavily package not available or Tavily import failed. Web search disabled.")

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

    # -------------------------
    # File processing / parsing
    # -------------------------
    def process_files(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes a list of uploaded files and returns structured metadata."""
        processed_data = {}
        for file_data in files:
            filename = file_data['filename']
            content = file_data['content']
            ext = Path(filename).suffix.lower()
            logger.info(f"Processing file: {filename} (type: {ext})")

            try:
                if ext in ['.csv', '.tsv']:
                    df = pd.read_csv(BytesIO(content), sep=',' if ext == '.csv' else '\t')
                    processed_data[filename] = {'type': 'dataframe', 'data': df.to_dict('records'), 'columns': list(df.columns)}
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
                    text = "".join(page.extract_text() or "" for page in reader.pages)
                    processed_data[filename] = {'type': 'pdf', 'text': text}
                elif ext in ['.png', '.jpg', '.jpeg']:
                    processed_data[filename] = {
                        'type': 'image',
                        'base64': base64.b64encode(content).decode('utf-8')
                    }
                else:  # Default to text
                    processed_data[filename] = {'type': 'text', 'content': content.decode('utf-8', errors='ignore')}
            except Exception as e:
                logger.error(f"Failed to process file {filename}: {e}")
                processed_data[filename] = {'type': 'error', 'error': str(e)}
        return processed_data

    # -------------------------
    # Visualization helpers
    # -------------------------
    def _encode_plot_to_base64_with_limit(self, fig, max_bytes: int = 100 * 1024) -> Optional[str]:
        """Encode a matplotlib figure to base64 PNG and attempt to keep it under max_bytes."""
        try:
            def _encode(dpi_val, quality='high'):
                buf = BytesIO()
                if quality == 'high':
                    fig.savefig(buf, format='png', dpi=dpi_val, bbox_inches='tight')
                else:
                    fig.savefig(buf, format='png', dpi=dpi_val, bbox_inches='tight', facecolor='white', edgecolor='none')
                buf.seek(0)
                return buf.getvalue()

            dpi = 120
            img_data = _encode(dpi)
            while len(img_data) > max_bytes and dpi > 30:
                dpi = int(dpi * 0.8)
                img_data = _encode(dpi)
            if len(img_data) > max_bytes:
                # final attempt: lower quality
                dpi = 60
                img_data = _encode(dpi, quality='low')
            plt.close(fig)
            return base64.b64encode(img_data).decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to encode plot to base64 with size limit: {e}")
            try:
                plt.close(fig)
            except Exception:
                pass
            return None

    def _is_visualization_key(self, key: str) -> bool:
        """Heuristic: detect if a key is supposed to be a visualization."""
        key = (key or "").lower()
        patterns = ["graph", "chart", "plot", "histogram", "heatmap", "visual", "image"]
        return any(p in key for p in patterns)

    def _build_network_graph(self, df: pd.DataFrame) -> Optional[str]:
        """Attempt to build a network graph (networkx) from a two-column edge list DataFrame."""
        try:
            # Choose likely source/target columns
            cols = list(df.columns)
            if len(cols) >= 2:
                source_col, target_col = cols[0], cols[1]
            else:
                logger.warning("Not enough columns to build a network graph.")
                return None

            G = nx.from_pandas_edgelist(df, source=source_col, target=target_col)
            fig, ax = plt.subplots(figsize=(6, 6))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=800, ax=ax, font_size=9)
            return self._encode_plot_to_base64_with_limit(fig)
        except Exception as e:
            logger.error(f"Failed to build network graph: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _build_histogram(self, df: pd.DataFrame, color="green") -> Optional[str]:
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            # choose a numeric column if available
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                ax.hist(df[col].dropna(), bins=20, color=color, alpha=0.8, edgecolor='black')
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                ax.set_title(f"Histogram of {col}")
            else:
                # fallback: histogram of first column's value counts
                ser = df.iloc[:, 0].value_counts()
                ax.bar(ser.index.astype(str), ser.values, color=color)
                ax.set_xlabel(df.columns[0])
                ax.set_ylabel("Count")
                ax.set_title(f"Value counts for {df.columns[0]}")
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            return self._encode_plot_to_base64_with_limit(fig)
        except Exception as e:
            logger.error(f"Failed to build histogram: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _build_scatter(self, df: pd.DataFrame) -> Optional[str]:
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            # Prefer numeric columns for x and y
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                ax.scatter(df[x_col].dropna(), df[y_col].dropna(), alpha=0.7)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"Scatter: {x_col} vs {y_col}")
            elif df.shape[1] >= 2:
                # use first two columns if numeric conversion possible
                ax.scatter(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str), alpha=0.7)
            else:
                # single column - plot index vs values
                ax.plot(df.iloc[:, 0])
                ax.set_title(f"Line plot of {df.columns[0]}")
            plt.tight_layout()
            return self._encode_plot_to_base64_with_limit(fig)
        except Exception as e:
            logger.error(f"Failed to build scatter: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _fill_visualizations(self, result: dict, df_map: Dict[str, pd.DataFrame]) -> dict:
        """
        Fill in missing visualization keys with auto-generated charts.
        df_map: {filename: DataFrame}
        """
        # Convenience: choose best DataFrame for a given key
        def choose_df_for_key(key: str) -> Optional[pd.DataFrame]:
            key_lower = key.lower()
            # try to match filename mentions
            for fname, df in df_map.items():
                if fname.lower() in key_lower:
                    return df
            # heuristics: if key mentions 'edge' or 'nodes' prefer files with two columns
            if "edge" in key_lower or "node" in key_lower or "graph" in key_lower or "network" in key_lower:
                for df in df_map.values():
                    if df.shape[1] >= 2:
                        return df
            # default: return largest dataframe (most rows)
            best = None
            best_rows = -1
            for df in df_map.values():
                try:
                    r = len(df)
                    if r > best_rows:
                        best_rows = r
                        best = df
                except Exception:
                    continue
            return best

        # Iterate and fill
        for key, value in list(result.items()):
            if (value is None or (isinstance(value, str) and value.lower() == "null")) and self._is_visualization_key(key):
                try:
                    df = choose_df_for_key(key)
                    if df is None:
                        result[key] = None
                        continue
                    if "network" in key.lower() or "graph" in key.lower():
                        result[key] = self._build_network_graph(df)
                    elif "histogram" in key.lower():
                        # optionally detect color request in key (e.g., 'degree_histogram_green'), but default green
                        result[key] = self._build_histogram(df, color="green")
                    else:
                        # generic fallback (scatter / histogram)
                        # if any numeric columns -> scatter, else histogram
                        if len(df.select_dtypes(include=[np.number]).columns) >= 2:
                            result[key] = self._build_scatter(df)
                        else:
                            result[key] = self._build_histogram(df)
                except Exception as e:
                    logger.error(f"Error while filling visualization for key '{key}': {e}")
                    logger.debug(traceback.format_exc())
                    result[key] = None
        return result

    # -------------------------
    # LLM / prompt generation
    # -------------------------
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
        prompt_parts.append("1. Analyze all the provided data to answer the user's questions.")
        prompt_parts.append("2. Perform all necessary calculations yourself, such as counting, date differences, and regression analysis, using the provided data.")
        prompt_parts.append("3. If a user asks for a plot or visualization, do NOT generate the image data yourself. Instead, include a key named 'visualization_request' in your JSON response. The value should be another JSON object specifying the 'type' (e.g., 'scatter'), 'x' column, 'y' column, and an optional 'title'. The columns MUST exist in the provided data. Example: \"visualization_request\": {\"type\": \"scatter\", \"x\": \"year\", \"y\": \"days_of_delay\", \"title\": \"Year vs. Delay\"}}")
        prompt_parts.append("4. Your final output MUST be a single, valid JSON object.")
        prompt_parts.append("⚠️ Important: Always include a full visualization_request object with 'type', 'x', 'y', and 'title'. Never just return the string 'visualization_request'.")

        return "\n".join(prompt_parts)

    # -------------------------
    # Main LLM response handling
    # -------------------------
    def generate_analysis_response(self, questions: str, context: Dict[str, Any]) -> Any:
        """Generates the main analysis response using the Gemini model, then fills visualizations if missing."""
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
                # Return whatever text we got
                return {"analysis_text": cleaned_text}

            # Build df_map from context (filename -> DataFrame) for fallback visualizations
            df_map: Dict[str, pd.DataFrame] = {}
            # duckdb result
            if context.get('duckdb_query_result'):
                try:
                    df = pd.DataFrame(context['duckdb_query_result']['data'])
                    df_map['duckdb_query_result'] = df
                except Exception:
                    pass
            # file_data
            if context.get('file_data'):
                for fname, meta in context['file_data'].items():
                    try:
                        if meta.get('type') == 'dataframe':
                            df_map[fname] = pd.DataFrame(meta['data'])
                        elif meta.get('type') == 'excel':
                            # take first sheet
                            sheets = meta.get('sheets', {})
                            if sheets:
                                first_sheet = next(iter(sheets.values()))
                                df_map[fname] = pd.DataFrame(first_sheet)
                    except Exception:
                        logger.debug(f"Failed to rebuild dataframe for file {fname}", exc_info=True)
            # scraped_data: take first table found
            if context.get('scraped_data'):
                for url, sdata in context['scraped_data'].items():
                    try:
                        tables = sdata.get('tables') or []
                        if tables:
                            df_map[f"scraped:{url}"] = pd.DataFrame(tables[0])
                    except Exception:
                        continue

            # Fill missing visualizations heuristically
            try:
                result = self._fill_visualizations(result, df_map)
            except Exception as e:
                logger.warning(f"Visualization filling failed: {e}")
                logger.debug(traceback.format_exc())

            # Now post-process any explicit visualization_request objects the LLM provided
            def process_visuals(obj, parent_key=None):
                """Recursively find and replace visualization requests with base64 strings."""
                if isinstance(obj, dict):
                    processed_dict = {}
                    for k, v in obj.items():
                        # Check if this key suggests it's a visualization request
                        is_viz_key = any(viz_word in (k or "").lower() for viz_word in [
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
                                # If LLM didn't provide details, skip (we already attempted fallback above)
                                processed_dict[k] = processed_dict.get(k, None) or None
                            else:
                                processed_dict[k] = process_visuals(v, k)
                        else:
                            # Handle nested visualization_request objects
                            if k == "visualization_request" and isinstance(v, dict):
                                return self.create_visualization(v, context)
                            elif isinstance(v, dict) and "type" in v and any(chart_type in str(v.get("type", "")).lower() for chart_type in ["scatter", "bar", "line", "pie", "histogram", "graph"]):
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
            logger.error(traceback.format_exc())
            return {"error": f"An error occurred with the Gemini API: {e}"}

    # -------------------------
    # Create visualization from explicit request
    # -------------------------
    def create_visualization(self, viz_request: Dict, context: Dict) -> Optional[str]:
        """Generically create any type of plot requested by the LLM and return base64 PNG under 100kB."""
        if not viz_request or not isinstance(viz_request, dict):
            return None

        logger.info(f"Handling visualization request: {viz_request}")

        # --- Get a DataFrame from context (best effort) ---
        df = None
        try:
            if context.get('duckdb_query_result'):
                df = pd.DataFrame(context['duckdb_query_result']['data'])
        except Exception:
            df = None

        if df is None and context.get('file_data'):
            for file_info in context['file_data'].values():
                if file_info.get('type') == 'dataframe':
                    try:
                        df = pd.DataFrame(file_info['data'])
                        break
                    except Exception:
                        continue

        if df is None and context.get('scraped_data'):
            for url_data in context['scraped_data'].values():
                try:
                    if url_data.get('tables'):
                        df = pd.DataFrame(url_data['tables'][0])
                        break
                except Exception:
                    continue

        if df is None or df.empty:
            logger.warning("No usable dataframe found for visualization.")
            return None

        try:
            viz_type = str(viz_request.get("type", "")).lower()
            # Map 'graph'/'network' to network builder
            if viz_type in ["graph", "network", "network_graph", "network-graph"]:
                return self._build_network_graph(df)

            # Use histogram
            if viz_type in ["histogram", "bar_histogram", "degree_histogram"]:
                # allow explicit column name
                col = viz_request.get("column") or viz_request.get("x") or (df.select_dtypes(include=[np.number]).columns[0] if len(df.select_dtypes(include=[np.number]).columns) > 0 else df.columns[0])
                fig, ax = plt.subplots(figsize=(8, 5))
                try:
                    ax.hist(df[col].dropna(), bins=20, color=viz_request.get("color", "green"), alpha=0.8, edgecolor='black')
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
                    ax.set_title(viz_request.get("title", f"Histogram of {col}"))
                except Exception:
                    # fallback to value counts
                    ser = df.iloc[:, 0].value_counts()
                    ax.clear()
                    ax.bar(ser.index.astype(str), ser.values, color=viz_request.get("color", "green"))
                    ax.set_xlabel(df.columns[0])
                    ax.set_ylabel("Count")
                    ax.set_title(viz_request.get("title", "Value counts"))
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                return self._encode_plot_to_base64_with_limit(fig)

            # Scatter
            if viz_type == "scatter":
                x_col, y_col = viz_request.get("x"), viz_request.get("y")
                if x_col in df.columns and y_col in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(df[x_col], df[y_col], alpha=0.7, c=viz_request.get("color", "blue"))
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(viz_request.get("title", f"{y_col} vs {x_col}"))
                    return self._encode_plot_to_base64_with_limit(fig)
                else:
                    logger.error(f"Scatter plot: Missing columns {x_col}, {y_col}")
                    return None

            # Bar
            if viz_type == "bar":
                x_col, y_col = viz_request.get("x"), viz_request.get("y")
                if x_col in df.columns and y_col in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.bar(df[x_col], df[y_col], color=viz_request.get("color", "blue"), alpha=0.7)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(viz_request.get("title", "Bar Chart"))
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    return self._encode_plot_to_base64_with_limit(fig)
                else:
                    logger.error(f"Bar chart: Missing columns {x_col}, {y_col}")
                    return None

            # Line
            if viz_type == "line":
                x_col, y_col = viz_request.get("x"), viz_request.get("y")
                if x_col in df.columns and y_col in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(df[x_col], df[y_col], marker='o', linewidth=2)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(viz_request.get("title", "Line Chart"))
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    return self._encode_plot_to_base64_with_limit(fig)
                else:
                    logger.error(f"Line chart: Missing columns {x_col}, {y_col}")
                    return None

            # Pie
            if viz_type == "pie":
                labels_col, values_col = viz_request.get("labels"), viz_request.get("values")
                if labels_col in df.columns and values_col in df.columns:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(df[values_col], labels=df[labels_col], autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    return self._encode_plot_to_base64_with_limit(fig)
                else:
                    logger.error(f"Pie chart: Missing columns {labels_col}, {values_col}")
                    return None

            logger.error(f"Unsupported visualization type: {viz_type}")
            return None

        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            logger.debug(traceback.format_exc())
            return None

    # -------------------------
    # Search helper (tavily)
    # -------------------------
    def search_internet(self, query: str) -> Optional[str]:
        """Performs a web search using Tavily and returns a summarized string of results."""
        if not self.tavily_client:
            logger.warning("Tavily client not initialized. Skipping web search.")
            return None

        try:
            logger.info(f"Performing Tavily web search for: '{query}'")
            response = self.tavily_client.search(query=query, search_depth="advanced", max_results=5)
            results_str = "\n\n".join(
                [f"Title: {res['title']}\nURL: {res['url']}\nContent: {res['content']}" for res in response.get('results', [])]
            )
            return results_str
        except Exception as e:
            logger.error(f"Tavily web search failed: {e}")
            return "Web search failed to retrieve results."


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

            # Let the analyzer handle everything (it will parse files)
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

    if not config.tavily_api_key:
        print("\n⚠️  WARNING: TAVILY_API_KEY not found!", file=sys.stderr)
        print("Web search functionality will be disabled.", file=sys.stderr)

    print(f"""
    🤖 Universal Data Analyst Agent
    ===============================
    🌐 Server running at: http://{config.host}:{config.port}
    🔧 Debug mode: {config.debug}
    🔍 LLM Model: {config.model_name}

    Press Ctrl+C to stop
    """)

    app.run(host=config.host, port=config.port, debug=config.debug)
