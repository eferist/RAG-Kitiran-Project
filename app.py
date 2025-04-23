# app.py
import os
import sys
import traceback
import torch  # <--- NEW: Import torch
from flask import Flask, request, jsonify, render_template, g
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the new service and pipeline classes
from src.llm_service import LLMService
from src.embedding_service import EmbeddingService
from src.vector_db import VectorDB
from src.reranker_service import RerankerService # <--- NEW: Import RerankerService
from src.query_pipeline import QueryPipeline

# --- Load Configuration from Environment Variables ---
# Consider moving this to a dedicated config module/class later
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT", "8080")
WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "KitiranDocsQA") # Use consistent name
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", 50)) # <-- NEW: Renamed & potentially updated default
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 5))        # <-- NEW: How many results after reranking
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", 'BAAI/bge-reranker-base') # <-- NEW: Reranker model
# --- End Configuration Loading ---

app = Flask(__name__)

# Global variable to hold the initialized QueryPipeline
query_pipeline_instance: QueryPipeline | None = None

def initialize_services():
    """Initializes all necessary services and the QueryPipeline."""
    global query_pipeline_instance
    print("--- Initializing Services for Flask App ---")
    vector_db = None # For finally block
    try:
        llm_service = LLMService()
        embedding_service = EmbeddingService()
        vector_db = VectorDB(host=WEAVIATE_HOST, port=WEAVIATE_PORT)

        # Check if collection exists before creating QueryPipeline
        if not vector_db.collection_exists(WEAVIATE_COLLECTION_NAME):
             print(f"ERROR: Collection '{WEAVIATE_COLLECTION_NAME}' does not exist.")
             print("Please run the main script in 'index' mode first.")
             query_pipeline_instance = None
             if vector_db:
                 vector_db.close()
             return False # Indicate failure

        # <--- NEW: Instantiate Reranker Service --->
        reranker_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {reranker_device} for Reranker.")
        try:
            reranker_service = RerankerService(
                model_name=RERANKER_MODEL_NAME,
                device=reranker_device
            )
        except Exception as e:
             print(f"FATAL ERROR during Reranker initialization: {e}")
             query_pipeline_instance = None
             if vector_db:
                 vector_db.close()
             return False # Indicate failure
        # <--- End Reranker Instantiation --->

        # <--- MODIFIED: Instantiate Query Pipeline --->
        query_pipeline_instance = QueryPipeline(
            llm_service=llm_service,
            embedding_service=embedding_service,
            vector_db=vector_db, # Pass the connected VectorDB instance
            reranker_service=reranker_service,         # Pass the reranker service instance
            collection_name=WEAVIATE_COLLECTION_NAME,
            retrieve_top_k=RETRIEVE_TOP_K,             # Use the renamed parameter
            rerank_top_n=RERANK_TOP_N                  # Pass the new parameter
        )
        # <--- End Pipeline Instantiation --->

        print("--- Services Initialized Successfully ---")
        return True # Indicate success

    except (ValueError, ConnectionError, RuntimeError) as e:
        print(f"FATAL ERROR during service initialization: {e}")
        traceback.print_exc()
        query_pipeline_instance = None
        if vector_db:
            vector_db.close() # Attempt to close DB if connection was partial
        return False # Indicate failure
    except Exception as e: # Catch any other unexpected errors during init
        print(f"UNEXPECTED FATAL ERROR during service initialization: {e}")
        traceback.print_exc()
        query_pipeline_instance = None
        if vector_db:
            vector_db.close()
        return False

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages from the frontend using the QueryPipeline."""
    global query_pipeline_instance

    if not query_pipeline_instance:
        print("Error: Query pipeline not initialized. Check logs.")
        return jsonify({"error": "Chat service is not available due to an initialization error. Please check server logs or ensure indexing was run."}), 503 # Service Unavailable

    try:
        data = request.get_json()
        user_message = data.get('message')
        history = data.get('history', []) # Allow passing history from frontend

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        print(f"Received message: {user_message[:50]}...")

        # Process the query using the initialized pipeline
        answer = query_pipeline_instance.process_query(
            query=user_message,
            history=history
        )

        print(f"Generated answer: {answer[:100]}...")
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error processing chat message: {e}")
        traceback.print_exc() # Log the full traceback for debugging
        return jsonify({"error": "An internal error occurred while processing your message."}), 500

# Optional: Add a shutdown hook - Note: Handling cleanup of non-request-scoped resources
# like the VectorDB client or models loaded in services on Flask shutdown can be tricky.
# Simple approaches might use `atexit` or signal handlers if running directly (not via Gunicorn/uWSGI).
# For robustness, consider managing resource lifecycles more explicitly.
# @app.teardown_appcontext
# def teardown_db(exception=None):
#     pass # Current setup keeps vector_db within initialize_services scope

if __name__ == '__main__':
    if initialize_services():
        # Use debug=True only for development
        # Set debug=False for production
        # Run on 0.0.0.0 to make it accessible on your network
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to initialize services. Flask application will not start.")
        sys.exit(1)