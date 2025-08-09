# test_client.py

import requests
import os
import json
import logging

# Configure basic logging for the client
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test():
    """Creates test files, sends a request to the API, and processes the response."""
    
    # --- 1. Prepare Test Files ---
    logging.info("Creating temporary test files (questions.txt, sample_data.csv)...")
    
    # Create a dummy CSV file
    csv_data = """id,name,age,salary
1,Alice,30,70000
2,Bob,25,60000
3,Charlie,35,80000
4,David,40,90000
5,Eve,28,65000
"""
    with open("sample_data.csv", "w") as f:
        f.write(csv_data)

    # Create a questions file that triggers different functionalities
    questions = """1. How many entries are in the provided dataset?
2. Create a scatterplot of age vs. salary.
"""
    with open("questions.txt", "w") as f:
        f.write(questions)

    # --- 2. Send Request to the API ---
    api_url = "http://localhost:8000/api/"
    
    # Prepare the files for the multipart/form-data request
    try:
        with open('questions.txt', 'rb') as q_file, open('sample_data.csv', 'rb') as d_file:
            files_to_upload = [
                ('files', ('questions.txt', q_file, 'text/plain')),
                ('files', ('sample_data.csv', d_file, 'text/csv'))
            ]

            logging.info(f"🚀 Sending POST request to {api_url}...")
            response = requests.post(api_url, files=files_to_upload, timeout=60)

        # --- 3. Process the Response ---
        logging.info(f"✅ Request completed with status code: {response.status_code}")

        if response.status_code == 200:
            logging.info("Response JSON received:")
            try:
                # Pretty print the JSON response
                response_json = response.json()
                print(json.dumps(response_json, indent=2))
                
                # Basic validation of the response
                if isinstance(response_json, list) and len(response_json) == 2:
                    logging.info("PASS: Response is a list with 2 items as expected.")
                else:
                    logging.warning("FAIL: Response format is not an array of 2 items.")

            except json.JSONDecodeError:
                logging.error("FAIL: The response from the server is not valid JSON.")
                print(response.text)
        else:
            logging.error(f"FAIL: The server returned an error. Response content:")
            print(response.text)

    except requests.exceptions.ConnectionError:
        logging.error("❌ Connection Error: Could not connect to the server.")
        logging.error("   Please make sure the FastAPI application is running on http://localhost:8000")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during the request: {e}")
        return
    finally:
        logging.info("🧹 Cleaning up test files...")
        if os.path.exists("sample_data.csv"):
            os.remove("sample_data.csv")
        if os.path.exists("questions.txt"):
            os.remove("questions.txt")
        logging.info("Cleanup complete.")

if __name__ == "__main__":
    run_test()
