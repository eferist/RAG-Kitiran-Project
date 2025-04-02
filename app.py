# app.py
import os
import sys
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import your RAG modules
from src import embedding_model
from src import vector_store
from src import llm # Assuming llm.py has the generate_answer function

# --- Load Configuration from Environment Variables ---
# Re-load necessary configs here as the server runs independently
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "bert-base-uncased")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT", "8080")
WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "DefaultCollection")
WEAVIATE_TOP_K = int(os.getenv("WEAVIATE_TOP_K", 5))
# --- End Configuration Loading ---

app = Flask(__name__)

# Global variable for Weaviate client (initialize later)
weaviate_client = None
weaviate_collection = None

def initialize_weaviate():
    """Connects to Weaviate and gets the collection."""
    global weaviate_client, weaviate_collection
    try:
        print(f"Connecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}...")
        weaviate_client = vector_store.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
        print("Connected to Weaviate.")

        collection_name = WEAVIATE_COLLECTION_NAME
        if weaviate_client.collections.exists(collection_name):
            weaviate_collection = weaviate_client.collections.get(collection_name)
            print(f"Using existing collection: '{collection_name}'")
        else:
            print(f"Error: Collection '{collection_name}' does not exist. Please run indexing first.")
            # In a real app, you might handle this more gracefully
            weaviate_collection = None

    except Exception as e:
        print(f"Error connecting to Weaviate or getting collection: {e}")
        weaviate_client = None
        weaviate_collection = None

@app.route('/')
def index():
    """Serves the main HTML page."""
    # Flask will look for 'index.html' in a 'templates' folder
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages from the frontend."""
    global weaviate_collection # Use the global collection object

    if not weaviate_collection:
        return jsonify({"error": "Weaviate collection not available. Please ensure indexing is complete and the server is connected."}), 500

    try:
        data = request.get_json()
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        print(f"Received message: {user_message}")

        # --- RAG Implementation ---
        print(f"Creating query embedding using model {EMBEDDING_MODEL_NAME}...")
        query_embedding = embedding_model.create_query_embedding(user_message, model_name=EMBEDDING_MODEL_NAME)

        print(f"Retrieving top {WEAVIATE_TOP_K} similar QA pairs based on question similarity...")
        # get_similar_chunks returns objects whose *question* vectors matched the query vector
        similar_objects = vector_store.get_similar_chunks(weaviate_collection, query_embedding, top_k=WEAVIATE_TOP_K)
        # Extract the *answer* from the properties of the retrieved objects to use as context
        context = "\n---\n".join([obj.properties["answer"] for obj in similar_objects if "answer" in obj.properties])
        print(f"Retrieved answers snippet for context: {context[:200]}...") # Log snippet of answers

        # --- LLM Call ---
        print("Generating answer...")
        # Assuming llm.generate_answer can handle None context if retrieval fails,
        # and potentially a history if we add it later. For now, no history.
        # Modify llm.generate_answer if it requires different parameters.
        # Pass an empty list for history to satisfy the function signature
        answer = llm.generate_answer(query=user_message, context=context, history=[])

        if answer:
            print(f"Generated answer: {answer}")
            return jsonify({"answer": answer})
        else:
            print("LLM failed to generate an answer.")
            return jsonify({"error": "Failed to generate response from LLM"}), 500

    except Exception as e:
        print(f"Error processing chat message: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred"}), 500

if __name__ == '__main__':
    initialize_weaviate() # Connect to Weaviate when the server starts
    # Use debug=True for development, it enables auto-reloading
    app.run(debug=True, port=5000) # Runs on http://127.0.0.1:5000/