import sys
sys.path.append(".")
from src import embedding_model
from src import vector_store
import weaviate

def main():
    # Connect to Weaviate
    try:
        client = vector_store.connect_to_local()
        print("Connected to Weaviate.")
    except Exception as e:
        print(f"Failed to connect to Weaviate: {e}")
        print("Please ensure Weaviate is running (e.g., via 'docker-compose up -d').")
        return

    # Check if the collection exists
    collection_name = "Question"
    if not client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' does not exist.")
        print("Please run the main script (src/main.py) first to create and populate the collection.")
        client.close()
        return

    collection = client.collections.get(collection_name)
    print(f"Using collection: '{collection_name}'")

    try:
        # Get the query from the user
        query = input("Enter your query: ")

        if not query:
            print("No query entered.")
            return

        # Create the query embedding
        print("Creating query embedding...")
        query_embedding = embedding_model.create_query_embedding(query)
        print("Query embedding created.")

        # Get the similar chunks (adjust top_k if needed)
        top_k = 3 # How many chunks to retrieve
        print(f"Searching for top {top_k} similar chunks...")
        similar_chunks_response = vector_store.get_similar_chunks(collection, query_embedding, top_k=top_k)

        # Print the retrieved chunks
        if similar_chunks_response:
            print(f"\n--- Top {len(similar_chunks_response)} Retrieved Chunks ---")
            for i, obj in enumerate(similar_chunks_response):
                print(f"Chunk {i+1} (UUID: {obj.uuid}):")
                print(obj.properties.get("content", "N/A"))
                # Optionally print score/distance if available and needed
                # print(f"Score/Distance: {obj.metadata...}") # Adjust based on Weaviate client version
                print("-" * 20)
        else:
            print("No similar chunks found.")

    except Exception as e:
        print(f"An error occurred during query processing: {e}")
    finally:
        # Close the client connection
        client.close()
        print("Weaviate connection closed.")


if __name__ == "__main__":
    main()