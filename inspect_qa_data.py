import os
import sys
from dotenv import load_dotenv
import weaviate
import weaviate.classes as wvc

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path if this script is run from the root
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import vector_store module after potentially adding src to path
try:
    from src import vector_store
except ImportError:
    print("Error: Could not import src.vector_store. Make sure the script is run from the project root or src is in PYTHONPATH.")
    sys.exit(1)


# --- Load Configuration from Environment Variables ---
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT", "8080")
WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "DefaultCollection") # Use the same name as indexing
# --- End Configuration Loading ---

def inspect_data():
    """Connects to Weaviate, fetches QA data, and prints it."""
    client = None
    collection_name = WEAVIATE_COLLECTION_NAME

    try:
        print(f"Connecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}...")
        # Use the connect function from vector_store
        client = vector_store.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
        print("Connected to Weaviate.")

        if not client.collections.exists(collection_name):
            print(f"Error: Collection '{collection_name}' does not exist.")
            return

        collection = client.collections.get(collection_name)
        print(f"Fetching data from collection: '{collection_name}'...")

        # Fetch objects - use a large limit to get all/most data for inspection
        # Adjust limit if you expect significantly more than 10000 items
        response = collection.query.fetch_objects(limit=10000)

        print(f"\n--- Found {len(response.objects)} QA Objects ---")
        if not response.objects:
            print("No data found in the collection.")
            return

        for i, obj in enumerate(response.objects):
            print(f"\n--- Object {i+1} ---")
            question = obj.properties.get("question", "N/A")
            answer = obj.properties.get("answer", "N/A")
            source_chunk = obj.properties.get("source_chunk", "N/A")

            print(f"  Question: {question}")
            print(f"  Answer: {answer}")
            print(f"  Source Chunk: {source_chunk[:150]}...") # Print snippet of source

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if client:
            client.close()
            print("\nWeaviate connection closed.")

if __name__ == "__main__":
    inspect_data()