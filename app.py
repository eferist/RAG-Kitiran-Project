# app.py
import os
import sys
import traceback
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
from src.query_pipeline import QueryPipeline

# --- Load Configuration from Environment Variables ---
# Consider moving this to a dedicated config module/class later
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT", "8080")
WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "KitiranDocsQA") # Use consistent name
WEAVIATE_TOP_K = int(os.getenv("WEAVIATE_TOP_K", 5))
# --- End Configuration Loading ---

app = Flask(__name__)

# Global variable to hold the initialized QueryPipeline
# Using Flask's 'g' object is often preferred for request-scoped resources,
# but for a singleton service like this, a global or app context might be okay.
# For simplicity here, we'll use a global initialized at startup.
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
             # Decide how to handle this: raise error, disable chat, etc.
             # For now, we'll let the chat endpoint fail if pipeline is None.
             query_pipeline_instance = None
             # Optionally close DB connection if we can't proceed
             if vector_db:
                 vector_db.close()
             return False # Indicate failure

        query_pipeline_instance = QueryPipeline(
            llm_service=llm_service,
            embedding_service=embedding_service,
            vector_db=vector_db, # Pass the connected VectorDB instance
            collection_name=WEAVIATE_COLLECTION_NAME,
            top_k=WEAVIATE_TOP_K
        )
        print("--- Services Initialized Successfully ---")
        return True # Indicate success

    except (ValueError, ConnectionError, RuntimeError) as e:
        print(f"FATAL ERROR during service initialization: {e}")
        query_pipeline_instance = None
        if vector_db:
            vector_db.close() # Attempt to close DB if connection was partial
        return False # Indicate failure
    # Note: We don't close the vector_db connection here if successful,
    # as the QueryPipeline needs it. It should be closed on app shutdown ideally.

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
        # TODO: Implement proper conversation history handling if needed
        # For now, passing empty history like original app.py and main.py query mode
        history = data.get('history', []) # Allow passing history from frontend if desired

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

# Optional: Add a shutdown hook to close the Weaviate client gracefully
@app.teardown_appcontext
def teardown_db(exception=None):
    # This might not be the perfect place if vector_db is not request-scoped
    # A more robust solution might involve signal handling or atexit
    # For now, this demonstrates the idea, but initialize_services() doesn't use 'g'
    pass
    # db = g.pop('vector_db', None)
    # if db is not None:
    #     db.close()

if __name__ == '__main__':
    if initialize_services():
        # Use debug=True only for development, it enables auto-reloading
        # Set debug=False for production
        app.run(host='0.0.0.0', port=5000, debug=False) # Run on port 5000, accessible externally
    else:
        print("Failed to initialize services. Flask application will not start.")
        # Exit or prevent app.run()
        sys.exit(1)