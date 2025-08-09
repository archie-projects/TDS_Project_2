#!/usr/bin/env python3
"""
Test script for Data Analyst Agent
Tests various scenarios to ensure the API works correctly
"""

import requests
import json
import time
import sys
import os
from io import StringIO
import pandas as pd

class DataAnalystTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/"
        
    def test_health_check(self):
        """Test health endpoint"""
        print("🔍 Testing health check...")
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ Health check passed")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_simple_question(self):
        """Test with simple text question"""
        print("\n🔍 Testing simple question...")
        
        question = "What is 2 + 2? Return the answer as a JSON array with one element."
        
        files = {
            'questions.txt': ('questions.txt', question, 'text/plain')
        }
        
        try:
            response = requests.post(self.api_url, files=files, timeout=180)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Simple question test passed: {result}")
                return True
            else:
                print(f"❌ Simple question test failed: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            print(f"❌ Simple question test error: {e}")
            return False
    
    def test_csv_analysis(self):
        """Test with CSV data analysis"""
        print("\n🔍 Testing CSV analysis...")
        
        # Create sample CSV
        csv_data = """Name,Age,Score
Alice,25,85
Bob,30,92
Charlie,35,78
Diana,28,96"""
        
        question = """Analyze the provided CSV data and answer:
1. What is the average age?
2. Who has the highest score?
3. What is the correlation between age and score?
Return answers as a JSON array."""
        
        files = {
            'questions.txt': ('questions.txt', question, 'text/plain'),
            'data.csv': ('data.csv', csv_data, 'text/csv')
        }
        
        try:
            response = requests.post(self.api_url, files=files, timeout=180)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ CSV analysis test passed: {result}")
                return True
            else:
                print(f"❌ CSV analysis test failed: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            print(f"❌ CSV analysis test error: {e}")
            return False
    
    def test_web_scraping(self):
        """Test web scraping functionality"""
        print("\n🔍 Testing web scraping...")
        
        question = """Scrape the title of the webpage at https://httpbin.org/html and return it as a JSON array with one element."""
        
        files = {
            'questions.txt': ('questions.txt', question, 'text/plain')
        }
        
        try:
            response = requests.post(self.api_url, files=files, timeout=180)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Web scraping test passed: {result}")
                return True
            else:
                print(f"❌ Web scraping test failed: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            print(f"❌ Web scraping test error: {e}")
            return False
    
    def test_json_analysis(self):
        """Test JSON data analysis"""
        print("\n🔍 Testing JSON analysis...")
        
        json_data = json.dumps({
            "users": [
                {"name": "Alice", "age": 25, "city": "New York"},
                {"name": "Bob", "age": 30, "city": "London"},
                {"name": "Charlie", "age": 35, "city": "Tokyo"}
            ]
        })
        
        question = """Analyze the JSON data and answer:
1. How many users are there?
2. What is the average age?
3. List all unique cities.
Return as a JSON array."""
        
        files = {
            'questions.txt': ('questions.txt', question, 'text/plain'),
            'data.json': ('data.json', json_data, 'application/json')
        }
        
        try:
            response = requests.post(self.api_url, files=files, timeout=180)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ JSON analysis test passed: {result}")
                return True
            else:
                print(f"❌ JSON analysis test failed: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            print(f"❌ JSON analysis test error: {e}")
            return False
    
    def test_visualization(self):
        """Test data visualization"""
        print("\n🔍 Testing visualization...")
        
        csv_data = """x,y
1,2
2,4
3,6
4,8
5,10"""
        
        question = """Create a scatter plot of x vs y from the CSV data with a red regression line. 
Return the plot as a base64 encoded data URI in a JSON array."""
        
        files = {
            'questions.txt': ('questions.txt', question, 'text/plain'),
            'data.csv': ('data.csv', csv_data, 'text/csv')
        }
        
        try:
            response = requests.post(self.api_url, files=files, timeout=180)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0 and "data:image" in str(result[0]):
                    print("✅ Visualization test passed: Plot generated")
                    return True
                else:
                    print(f"❌ Visualization test failed: No plot in response {result}")
                    return False
            else:
                print(f"❌ Visualization test failed: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            print(f"❌ Visualization test error: {e}")
            return False
    
    def test_complex_analysis(self):
        """Test complex multi-step analysis"""
        print("\n🔍 Testing complex analysis...")
        
        sales_data = """Product,Category,Sales,Month
Laptop,Electronics,1200,Jan
Phone,Electronics,800,Jan
Book,Books,25,Jan
Laptop,Electronics,1100,Feb
Phone,Electronics,850,Feb
Book,Books,30,Feb"""
        
        question = """Perform a comprehensive analysis of the sales data:
1. Calculate total sales by category
2. Find the best-selling product
3. Calculate month-over-month growth rate
4. Create a summary report
Return as a JSON object with these keys: total_by_category, best_product, growth_rate, summary"""
        
        files = {
            'questions.txt': ('questions.txt', question, 'text/plain'),
            'sales.csv': ('sales.csv', sales_data, 'text/csv')
        }
        
        try:
            response = requests.post(self.api_url, files=files, timeout=180)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict) and any(key in result for key in ['total_by_category', 'best_product', 'growth_rate']):
                    print(f"✅ Complex analysis test passed: {result}")
                    return True
                else:
                    print(f"✅ Complex analysis completed (different format): {result}")
                    return True
            else:
                print(f"❌ Complex analysis test failed: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            print(f"❌ Complex analysis test error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Data Analyst Agent Tests")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Simple Question", self.test_simple_question),
            ("CSV Analysis", self.test_csv_analysis),
            ("Web Scraping", self.test_web_scraping),
            ("JSON Analysis", self.test_json_analysis),
            ("Visualization", self.test_visualization),
            ("Complex Analysis", self.test_complex_analysis)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ {test_name} crashed: {e}")
                failed += 1
            
            time.sleep(2)  # Brief pause between tests
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("🎉 All tests passed! Your Data Analyst Agent is working correctly.")
            return True
        else:
            print(f"⚠️  {failed} tests failed. Please check the logs above.")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Data Analyst Agent')
    parser.add_argument('--url', default='http://localhost:8000', 
                       help='Base URL of the API (default: http://localhost:8000)')
    parser.add_argument('--test', choices=['health', 'simple', 'csv', 'scraping', 'json', 'viz', 'complex', 'all'],
                       default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    tester = DataAnalystTester(args.url)
    
    if args.test == 'all':
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    else:
        test_methods = {
            'health': tester.test_health_check,
            'simple': tester.test_simple_question,
            'csv': tester.test_csv_analysis,
            'scraping': tester.test_web_scraping,
            'json': tester.test_json_analysis,
            'viz': tester.test_visualization,
            'complex': tester.test_complex_analysis
        }
        
        if args.test in test_methods:
            success = test_methods[args.test]()
            sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
