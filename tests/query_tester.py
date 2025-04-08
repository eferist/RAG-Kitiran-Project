import sys
import os # Added
from dotenv import load_dotenv # Added
sys.path.append(".")
from src import embedding_model
from src import vector_store
import weaviate

def main():
    load_dotenv() # Added: Load .env file

    # Get config from environment variables
    weaviate_host = os.getenv("WEAVIATE_HOST", "localhost")
    weaviate_port = os.getenv("WEAVIATE_PORT", "8080")
    collection_name = os.getenv("WEAVIATE_COLLECTION_NAME", "Question")
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME") # Added: Load model name
    top_k_str = os.getenv("WEAVIATE_TOP_K", "3")
    if not embedding_model_name: # Added: Check if model name is loaded
        print("Error: EMBEDDING_MODEL_NAME not found in .env file.")
        return
    try:
        top_k = int(top_k_str)
    except ValueError:
        print(f"Warning: Invalid WEAVIATE_TOP_K value '{top_k_str}'. Using default 3.")
        top_k = 3

    client = None # Initialize client to None for finally block
    try:
        # Connect to Weaviate using config
        try:
            # Modified: Use loaded host and port
            client = vector_store.connect_to_local(host=weaviate_host, port=weaviate_port)
            print(f"Connected to Weaviate at {weaviate_host}:{weaviate_port}.") # Updated print
        except Exception as e:
            print(f"Failed to connect to Weaviate: {e}")
            # Updated print to show actual connection attempt details
            print(f"Please ensure Weaviate is running and reachable at {weaviate_host}:{weaviate_port}.")
            print("(Check .env configuration and 'docker-compose up -d' status).")
            return # Exit if connection fails

        # Check if the collection exists
        if not client.collections.exists(collection_name):
            print(f"Collection '{collection_name}' (from .env) does not exist.") # Updated print
            print("Please run the main script (src/main.py or app.py) first to create and populate the collection.")
            return # Exit if collection doesn't exist

        collection = client.collections.get(collection_name)
        print(f"Using collection: '{collection_name}' (from .env)") # Updated print

        # --- Start Query Loop ---
        while True:
            try:
                # Get the query from the user
                query = input("\nEnter your query (or type 'quit' to exit): ")

                if not query:
                    print("No query entered.")
                    continue # Ask for input again

                if query.lower() == 'quit':
                    print("Exiting...")
                    break # Exit the loop

                # Create the query embedding
                print(f"Creating query embedding using model: {embedding_model_name}...")
                query_embedding = embedding_model.create_query_embedding(query, model_name=embedding_model_name)
                print("Query embedding created.")

                # Get the similar chunks
                print(f"Searching for top {top_k} (from .env) similar objects...")
                similar_chunks_response = vector_store.get_similar_chunks(collection, query_embedding, top_k=top_k)

                # Print the retrieved objects
                if similar_chunks_response:
                    print(f"\n--- Top {len(similar_chunks_response)} Retrieved Objects (Max {top_k}) ---")
                    for i, obj in enumerate(similar_chunks_response):
                        print(f"--- Object {i+1} (UUID: {obj.uuid}) ---")
                        question = obj.properties.get("question", "N/A (question property not found)")
                        answer = obj.properties.get("answer", "N/A (answer property not found)")
                        source_chunk = obj.properties.get("source_chunk", "N/A (source_chunk property not found)")
                        print(f"  Question: {question}")
                        print(f"  Answer:   {answer}")
                        print(f"  Source:   {source_chunk}")
                        # Optionally print score/distance
                        # print(f"  Score/Distance: {obj.metadata...}")
                else:
                    print("No similar objects found.")

            except Exception as e:
                print(f"An error occurred during query processing: {e}")
                # Decide whether to break or continue the loop on error
                # continue
        # --- End Query Loop ---

    finally:
        # Close the client connection if it was successfully created
        if client:
            client.close()
            print("Weaviate connection closed.")

# This section is now encompassed within the main try...finally block above


if __name__ == "__main__":
    main()