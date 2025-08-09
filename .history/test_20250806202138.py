import requests
import os
import json
import logging

# --- Configuration ---
# ❗️ This URL points to your local server.
API_URL = "http://127.0.0.1:8000/api/"

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_files():
    """Creates temporary files needed for the test request."""
    logging.info("Creating temporary test files...")
    
    questions_content = """
    Scrape the list of highest grossing films from Wikipedia. It is at the URL:
    https://en.wikipedia.org/wiki/List_of_highest-grossing_films

    Answer the following questions and respond with a JSON array of strings containing the answer.
    1. How many films grossed over $2 billion?
    2. Which is the earliest film on the list?
    3. What's the correlation between 'Rank' and 'Year'?
    4. Draw a scatterplot of 'Year' and 'Rank' with a dotted red regression line.
       Return as a base-64 encoded data URI.
    """
    with open("questions.txt", "w", encoding="utf-8") as f:
        f.write(questions_content)

def cleanup_test_files():
    """Removes the temporary files created for the test."""
    logging.info("Cleaning up test files...")
    if os.path.exists("questions.txt"):
        os.remove("questions.txt")

def run_test():
    """Runs the full test against the local API."""
    create_test_files()
    
    try:
        with open('questions.txt', 'rb') as q_file:
            files_to_upload = {
                'files': ('questions.txt', q_file, 'text/plain')
            }

            logging.info(f"🚀 Sending POST request to {API_URL}...")
            response = requests.post(API_URL, files=files_to_upload, timeout=240) # 4 minute timeout

        logging.info(f"✅ Request completed with status code: {response.status_code}")

        if response.status_code == 200:
            logging.info("Response JSON received:")
            try:
                response_json = response.json()
                print(json.dumps(response_json, indent=2))
                
                if isinstance(response_json, (list, dict)) and len(response_json) > 0:
                    logging.info("PASS: Response format appears correct.")
                else:
                    logging.warning("FAIL: Response format might be incorrect.")

            except json.JSONDecodeError:
                logging.error("FAIL: The server's response was not valid JSON.")
                print("--- Server Response Text ---")
                print(response.text)
        else:
            logging.error(f"FAIL: The server returned an error.")
            print("--- Server Response Text ---")
            print(response.text)

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Connection Error: Could not connect to the server.")
        logging.error(f"   Make sure your FastAPI server is running in another terminal.")
        logging.error(f"   Error details: {e}")
    finally:
        cleanup_test_files()

if __name__ == "__main__":
    run_test()
