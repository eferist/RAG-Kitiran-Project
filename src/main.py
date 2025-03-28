import sys
sys.path.append(".")
from src import document_loader
from src import text_splitter
from src import embedding_model
from src import vector_store
from src import llm

def main():
    # Load the document
    text = document_loader.load_document("data/Kitiran Dokumen Context.pdf")

    # Split the text into chunks
    chunks = text_splitter.split_text(text, chunk_size=150, chunk_overlap=0)

    # Create embeddings for the chunks
    embeddings = embedding_model.create_embeddings(chunks)

    client = None  # Initialize client to None
    try:
        # Connect to Weaviate
        print("Connecting to Weaviate...")
        client = vector_store.connect_to_local()
        print("Connected to Weaviate.")

        collection_name = "Question"
        # Check if the collection exists, create if not
        if not client.collections.exists(collection_name):
            print(f"Collection '{collection_name}' does not exist. Creating...")
            collection = vector_store.create_collection(collection_name)
            print(f"Collection '{collection_name}' created.")
            # Add the data to Weaviate only when creating the collection
            print("Adding data to Weaviate...")
            vector_store.add_data_to_weaviate(collection, chunks, embeddings)
            print("Data added to Weaviate.")
        else:
            collection = client.collections.get(collection_name)
            print(f"Using existing collection: '{collection_name}'")
            # Consider if you want to re-add data every time or only if collection is new
            # For now, data is only added when the collection is first created.

        # Get the query from the user
        query = input("Enter your query: ")

        # Create the query embedding
        query_embedding = embedding_model.create_query_embedding(query)

        # Get the similar chunks
        similar_chunks = vector_store.get_similar_chunks(collection, query_embedding)

        # Combine the content of the similar chunks
        context = "\n".join([obj.properties["content"] for obj in similar_chunks])

        # Generate the answer using the LLM (original single-turn call)
        answer = llm.generate_answer(query, context)

        # Print the answer
        print(f"Assistant: {answer}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure the client is closed if it was successfully opened
        if client:
            client.close()
            print("Weaviate connection closed.")

if __name__ == "__main__":
    main()