#!/usr/bin/env python3
"""
Simple startup script for the Universal Data Analyst Agent
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the main application
from app import create_app

def main():
    """Start the application"""
    
    # Check for required files
    if not os.path.exists('index.html'):
        print("❌ Error: index.html not found in current directory!")
        print("Please make sure index.html is in the same directory as this script.")
        sys.exit(1)
    
    # Check environment variables
    if not os.getenv('GEMINI_API_KEY'):
        print("\n⚠️  WARNING: GEMINI_API_KEY not found!")
        print("Please set your Gemini API key for full functionality:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        print("\nGet your API key from: https://aistudio.google.com/app/apikey")
        print("The application will start with limited functionality.\n")
    
    # Create and run app
    app = create_app()
    
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', '5000'))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    print(f"""
🤖 Universal Data Analyst Agent
===============================
🌐 Server: http://{host}:{port}
📊 Ready to analyze any data!
🔍 Powered by: Gemini 1.5 Flash
📝 Supports: CSV, Excel, JSON, PDF, Images, Web scraping

Press Ctrl+C to stop
""")
    
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
