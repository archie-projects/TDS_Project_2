#!/usr/bin/env python3
"""
Comprehensive Testing Script for Universal Data Analyst Agent

This script tests all major functionality of the data analyst agent
and provides detailed feedback for improvements.

Usage:
    python test_agent.py --url http://localhost:8000 --api-key YOUR_KEY
    python test_agent.py --run-all-tests
    python test_agent.py --create-test-data
"""

import asyncio
import aiofiles
import aiohttp
import json
import pandas as pd
import numpy as np
import argparse
import sys
import os
import time
import tempfile
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import base64
import matplotlib.pyplot as plt
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Colors:
    """ANSI color codes for pretty output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class TestDataGenerator:
    """Generate various types of test data"""
    
    @staticmethod
    def create_sales_data(rows: int = 100) -> pd.DataFrame:
        """Create sample sales dataset"""
        np.random.seed(42)
        regions = ['North', 'South', 'East', 'West']
        products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
        
        data = {
            'Date': pd.date_range('2023-01-01', periods=rows, freq='D'),
            'Region': np.random.choice(regions, rows),
            'Product': np.random.choice(products, rows),
            'Sales': np.random.normal(1000, 200, rows).round(2),
            'Profit': np.random.normal(150, 50, rows).round(2),
            'Units_Sold': np.random.poisson(10, rows),
            'Discount': np.random.uniform(0, 0.3, rows).round(3)
        }
        
        # Add some correlation between sales and profit
        df = pd.DataFrame(data)
        df['Profit'] = (df['Sales'] * 0.15 + np.random.normal(0, 20, rows)).round(2)
        
        return df
    
    @staticmethod
    def create_financial_data(rows: int = 50) -> pd.DataFrame:
        """Create sample financial dataset"""
        np.random.seed(123)
        companies = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        
        data = {
            'Company': np.random.choice(companies, rows),
            'Revenue': np.random.lognormal(10, 1, rows).round(2),
            'Expenses': np.random.lognormal(9.5, 1, rows).round(2),
            'Market_Cap': np.random.lognormal(12, 1, rows).round(2),
            'PE_Ratio': np.random.uniform(5, 50, rows).round(2),
            'Year': np.random.choice([2020, 2021, 2022, 2023], rows)
        }
        
        df = pd.DataFrame(data)
        df['Net_Income'] = (df['Revenue'] - df['Expenses']).round(2)
        
        return df
    
    @staticmethod
    async def create_test_files(output_dir: str) -> Dict[str, str]:
        """Create various test files"""
        files = {}
        
        # CSV file
        sales_df = TestDataGenerator.create_sales_data(150)
        csv_path = os.path.join(output_dir, 'sales_data.csv')
        sales_df.to_csv(csv_path, index=False)
        files['sales_csv'] = csv_path
        
        # Excel file
        financial_df = TestDataGenerator.create_financial_data(75)
        excel_path = os.path.join(output_dir, 'financial_data.xlsx')
        financial_df.to_excel(excel_path, index=False)
        files['financial_excel'] = excel_path
        
        # JSON file
        json_data = {
            'metadata': {
                'dataset_name': 'Sample Analytics Data',
                'created_date': '2024-01-01',
                'version': '1.0'
            },
            'metrics': [
                {'name': 'Total Revenue', 'value': 1500000, 'unit': 'USD'},
                {'name': 'Customer Count', 'value': 2500, 'unit': 'count'},
                {'name': 'Conversion Rate', 'value': 0.045, 'unit': 'percentage'}
            ],
            'regions': {
                'North': {'sales': 450000, 'customers': 800},
                'South': {'sales': 380000, 'customers': 650},
                'East': {'sales': 420000, 'customers': 720},
                'West': {'sales': 250000, 'customers': 330}
            }
        }
        json_path = os.path.join(output_dir, 'analytics_data.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        files['analytics_json'] = json_path
        
        # Simple image (for testing image handling)
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
        ax.set_xlabel('X values')
        ax.set_ylabel('Y values')
        ax.set_title('Sample Sine Wave')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        img_path = os.path.join(output_dir, 'sample_chart.png')
        plt.savefig(img_path, dpi=100, bbox_inches='tight')
        plt.close()
        files['sample_image'] = img_path
        
        return files

class DataAnalystTester:
    """Main testing class for the Data Analyst Agent"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/"
        self.session = None
        self.test_results = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300))
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"{Colors.OKGREEN}✓ Service is healthy{Colors.ENDC}")
                    return data
                else:
                    logger.error(f"{Colors.FAIL}✗ Health check failed: {response.status}{Colors.ENDC}")
                    return {"status": "unhealthy", "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"{Colors.FAIL}✗ Health check failed: {e}{Colors.ENDC}")
            return {"status": "error", "error": str(e)}
    
    async def submit_analysis(self, questions: str, files: Dict[str, str] = None) -> Tuple[bool, Dict[str, Any], float]:
        """Submit analysis request and return success, result, and time taken"""
        start_time = time.time()
        
        try:
            # Prepare multipart form data
            data = aiohttp.FormData()
            
            # Add questions as questions.txt
            data.add_field('files', questions, filename='questions.txt', content_type='text/plain')
            
            # Add files if provided
            if files:
                for file_key, file_path in files.items():
                    if os.path.exists(file_path):
                        async with aiofiles.open(file_path, 'rb') as f:
                            file_content = await f.read()
                            filename = os.path.basename(file_path)
                            # Determine content type
                            if filename.endswith('.csv'):
                                content_type = 'text/csv'
                            elif filename.endswith('.json'):
                                content_type = 'application/json'
                            elif filename.endswith(('.xlsx', '.xls')):
                                content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            elif filename.endswith('.png'):
                                content_type = 'image/png'
                            else:
                                content_type = 'application/octet-stream'
                            
                            data.add_field('files', file_content, filename=filename, content_type=content_type)
            
            async with self.session.post(self.api_url, data=data) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return True, result, response_time
                else:
                    error_text = await response.text()
                    return False, {"error": f"HTTP {response.status}: {error_text}"}, response_time
                    
        except Exception as e:
            response_time = time.time() - start_time
            return False, {"error": str(e)}, response_time
    
    async def run_test(self, test_name: str, questions: str, files: Dict[str, str] = None, 
                      expected_format: str = "json", validator_func=None) -> Dict[str, Any]:
        """Run a single test case"""
        logger.info(f"{Colors.OKBLUE}Running test: {test_name}{Colors.ENDC}")
        
        success, result, response_time = await self.submit_analysis(questions, files)
        
        test_result = {
            "test_name": test_name,
            "success": success,
            "response_time": response_time,
            "result": result,
            "questions": questions,
            "files_used": list(files.keys()) if files else [],
            "validation": {}
        }
        
        if success:
            # Validate result format
            if expected_format == "json_array" and not isinstance(result, list):
                test_result["validation"]["format_error"] = f"Expected JSON array, got {type(result)}"
            elif expected_format == "json_object" and not isinstance(result, dict):
                test_result["validation"]["format_error"] = f"Expected JSON object, got {type(result)}"
            
            # Custom validation
            if validator_func:
                try:
                    validation_result = validator_func(result)
                    test_result["validation"].update(validation_result)
                except Exception as e:
                    test_result["validation"]["validator_error"] = str(e)
            
            # Check for base64 images
            def check_for_images(obj, path=""):
                images_found = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        images_found.extend(check_for_images(value, f"{path}.{key}"))
                elif isinstance(obj, list):
                    for i, value in enumerate(obj):
                        images_found.extend(check_for_images(value, f"{path}[{i}]"))
                elif isinstance(obj, str) and obj.startswith("data:image/"):
                    size_kb = len(obj) / 1024
                    images_found.append({"path": path, "size_kb": round(size_kb, 1)})
                return images_found
            
            images = check_for_images(result)
            if images:
                test_result["validation"]["images_found"] = images
        
        self.test_results.append(test_result)
        
        # Print result
        if success:
            if test_result["validation"].get("format_error"):
                logger.warning(f"{Colors.WARNING}⚠ {test_name} - Format issue: {test_result['validation']['format_error']}{Colors.ENDC}")
            else:
                logger.info(f"{Colors.OKGREEN}✓ {test_name} - Success ({response_time:.2f}s){Colors.ENDC}")
        else:
            logger.error(f"{Colors.FAIL}✗ {test_name} - Failed: {result.get('error', 'Unknown error')}{Colors.ENDC}")
        
        return test_result
    
    async def run_comprehensive_tests(self, test_files: Dict[str, str]):
        """Run all comprehensive tests"""
        
        # Test 1: Simple data analysis
        await self.run_test(
            "Basic Data Analysis",
            """Analyze the uploaded CSV data:
1. How many total records are there?
2. What is the average sales value?
3. Which region has the highest total sales?

Return as JSON array: [record_count, average_sales, "region_name"]""",
            {"sales_data": test_files['sales_csv']},
            "json_array"
        )
        
        # Test 2: Statistical analysis
        await self.run_test(
            "Statistical Analysis",
            """Perform statistical analysis on the sales data:
{
    "correlation_sales_profit": "correlation coefficient",
    "total_revenue": "sum of all sales",
    "regression_slope": "slope of sales vs profit regression"
}""",
            {"sales_data": test_files['sales_csv']},
            "json_object",
            lambda r: {"has_correlation": "correlation_sales_profit" in r, "has_slope": "regression_slope" in r}
        )
        
        # Test 3: Visualization test
        await self.run_test(
            "Data Visualization",
            """Create a scatter plot of Sales vs Profit with a regression line.
Return the plot as: {"plot": "data:image/png;base64,...", "correlation": correlation_value}""",
            {"sales_data": test_files['sales_csv']},
            "json_object",
            lambda r: {
                "has_plot": "plot" in r and isinstance(r.get("plot"), str) and "data:image/" in r.get("plot", ""),
                "plot_size_ok": len(r.get("plot", "")) < 100000 if "plot" in r else False
            }
        )
        
        # Test 4: Multi-file analysis
        await self.run_test(
            "Multi-file Analysis",
            """Analyze both the sales CSV and financial Excel data:
1. How many records in each file?
2. What's the average revenue from the financial data?
3. Create a comparison chart.

Return as: [sales_records, financial_records, avg_revenue, "data:image/png;base64,..."]""",
            {"sales_data": test_files['sales_csv'], "financial_data": test_files['financial_excel']},
            "json_array"
        )
        
        # Test 5: JSON data analysis
        await self.run_test(
            "JSON Data Analysis",
            """Analyze the JSON analytics data:
{
    "total_customers": "sum of customers across all regions",
    "highest_sales_region": "region with highest sales",
    "average_sales_per_customer": "calculated metric"
}""",
            {"analytics_data": test_files['analytics_json']},
            "json_object"
        )
        
        # Test 6: Web scraping test (Wikipedia)
        await self.run_test(
            "Web Scraping Test",
            """Scrape data from Wikipedia's list of highest-grossing films:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer these questions:
1. How many films are listed in the main table?
2. What is the highest grossing film of all time?
3. How many films were released after 2010?

Return as JSON array: [total_films, "highest_grossing_film", films_after_2010]""",
            None,
            "json_array"
        )
        
        # Test 7: Complex aggregation
        await self.run_test(
            "Complex Data Aggregation",
            """Using the sales data, create a comprehensive analysis:
1. Group by Region and calculate total sales
2. Find the top 3 products by total units sold
3. Calculate month-over-month growth rate
4. Create a visualization showing regional performance

Return as:
{
    "regional_sales": {"North": value, "South": value, ...},
    "top_products": ["Product A", "Product B", "Product C"],
    "growth_analysis": "description",
    "regional_chart": "data:image/png;base64,..."
}""",
            {"sales_data": test_files['sales_csv']},
            "json_object"
        )
        
        # Test 8: Error handling
        await self.run_test(
            "Error Handling Test",
            """Analyze a non-existent column called 'InvalidColumn':
1. What is the average of InvalidColumn?
2. Create a chart of InvalidColumn vs Sales.

Handle this gracefully and return: {"error_handled": true, "message": "description"}""",
            {"sales_data": test_files['sales_csv']},
            "json_object"
        )
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for test in self.test_results if test["success"])
        
        report = f"""
{Colors.HEADER}{'='*60}
DATA ANALYST AGENT - COMPREHENSIVE TEST REPORT
{'='*60}{Colors.ENDC}

{Colors.BOLD}SUMMARY:{Colors.ENDC}
• Total Tests: {total_tests}
• Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)
• Failed: {total_tests - successful_tests}
• Average Response Time: {sum(test['response_time'] for test in self.test_results) / total_tests:.2f}s

{Colors.BOLD}DETAILED RESULTS:{Colors.ENDC}
"""
        
        for test in self.test_results:
            status = f"{Colors.OKGREEN}✓ PASS{Colors.ENDC}" if test["success"] else f"{Colors.FAIL}✗ FAIL{Colors.ENDC}"
            report += f"\n{status} {test['test_name']} ({test['response_time']:.2f}s)\n"
            
            if test["success"]:
                # Show validation results
                if test["validation"]:
                    for key, value in test["validation"].items():
                        if "error" in key.lower():
                            report += f"  {Colors.WARNING}⚠ {key}: {value}{Colors.ENDC}\n"
                        else:
                            report += f"  ✓ {key}: {value}\n"
                
                # Show result type and structure
                result_type = type(test["result"]).__name__
                if isinstance(test["result"], (list, dict)):
                    length = len(test["result"])
                    report += f"  📊 Result: {result_type} with {length} items\n"
                else:
                    report += f"  📊 Result: {result_type}\n"
                
                # Check for images
                if test["validation"].get("images_found"):
                    for img in test["validation"]["images_found"]:
                        report += f"  🖼️ Image found: {img['size_kb']}KB\n"
            else:
                error = test["result"].get("error", "Unknown error")
                report += f"  {Colors.FAIL}Error: {error[:100]}...{Colors.ENDC}\n"
        
        # Recommendations section
        report += f"\n{Colors.BOLD}RECOMMENDATIONS FOR IMPROVEMENT:{Colors.ENDC}\n"
        
        failed_tests = [test for test in self.test_results if not test["success"]]
        if failed_tests:
            report += f"• Fix {len(failed_tests)} failing tests\n"
        
        slow_tests = [test for test in self.test_results if test["response_time"] > 30]
        if slow_tests:
            report += f"• Optimize performance for {len(slow_tests)} slow tests (>30s)\n"
        
        format_issues = [test for test in self.test_results if test["validation"].get("format_error")]
        if format_issues:
            report += f"• Fix response format issues in {len(format_issues)} tests\n"
        
        tests_with_images = [test for test in self.test_results if test["validation"].get("images_found")]
        report += f"• Image generation working in {len(tests_with_images)} tests ✓\n"
        
        # Performance analysis
        avg_time = sum(test['response_time'] for test in self.test_results) / total_tests
        if avg_time > 60:
            report += "• Consider optimizing overall response time\n"
        
        report += f"\n{Colors.OKGREEN}Report generated at {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n"
        
        return report
    
    def save_detailed_results(self, filename: str = "test_results.json"):
        """Save detailed test results to JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_tests": len(self.test_results),
                "successful_tests": sum(1 for test in self.test_results if test["success"]),
                "tests": self.test_results
            }, f, indent=2, default=str)
        logger.info(f"Detailed results saved to {filename}")

async def main():
    parser = argparse.ArgumentParser(description="Test the Data Analyst Agent")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the agent")
    parser.add_argument("--create-test-data", action="store_true", help="Create test data files")
    parser.add_argument("--run-all-tests", action="store_true", help="Run all comprehensive tests")
    parser.add_argument("--output-dir", default="./test_data", help="Directory for test files")
    parser.add_argument("--save-results", default="test_results.json", help="Save results to file")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.create_test_data:
        logger.info("Creating test data files...")
        test_files = await TestDataGenerator.create_test_files(args.output_dir)
        logger.info(f"Created test files: {list(test_files.keys())}")
        return
    
    if args.run_all_tests:
        # Create test data first
        logger.info("Creating test data...")
        test_files = await TestDataGenerator.create_test_files(args.output_dir)
        
        # Run tests
        async with DataAnalystTester(args.url) as tester:
            # Health check first
            health = await tester.health_check()
            if health.get("status") != "healthy":
                logger.error("Service is not healthy. Aborting tests.")
                return
            
            logger.info(f"{Colors.HEADER}Starting comprehensive test suite...{Colors.ENDC}")
            await tester.run_comprehensive_tests(test_files)
            
            # Generate and display report
            report = tester.generate_report()
            print(report)
            
            # Save results
            if args.save_results:
                tester.save_detailed_results(args.save_results)
    
    else:
        print(f"""
{Colors.HEADER}Data Analyst Agent Testing Script{Colors.ENDC}

Available options:
  --create-test-data    Create sample test data files
  --run-all-tests      Run comprehensive test suite
  --url URL            Specify agent URL (default: http://localhost:8000)
  --output-dir DIR     Test data directory (default: ./test_data)

Examples:
  python test_agent.py --create-test-data
  python test_agent.py --run-all-tests
  python test_agent.py --run-all-tests --url http://your-app.com

Quick health check:
""")
        
        async with DataAnalystTester(args.url) as tester:
            health = await tester.health_check()
            print(f"Service status: {health}")

if __name__ == "__main__":
    asyncio.run(main())
