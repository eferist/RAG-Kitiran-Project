import sys
import argparse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append(".")
from src import document_loader
from src import text_splitter
from src import embedding_model
from src import vector_store
from src import llm

# --- Load Configuration from Environment Variables ---
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "data/default_document.pdf")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 0))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "bert-base-uncased")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost") # Corrected default based on previous fix
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT", "8080")
WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "DefaultCollection")
WEAVIATE_TOP_K = int(os.getenv("WEAVIATE_TOP_K", 5))
# --- End Configuration Loading ---

def main(args):
    client = None
    collection_name = WEAVIATE_COLLECTION_NAME

    try:
        print(f"Connecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}...")
        client = vector_store.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
        print("Connected to Weaviate.")

        # --- Indexing Mode ---
        if args.mode == 'index':
            # (Indexing logic remains unchanged)
            print("Running in INDEX mode.")
            if not client.collections.exists(collection_name):
                print(f"Collection '{collection_name}' does not exist. Creating and Indexing...")
                print(f"Loading document from {DOCUMENT_PATH}...")
                text = document_loader.load_document(DOCUMENT_PATH)
                print(f"Splitting text with chunk size {CHUNK_SIZE} and overlap {CHUNK_OVERLAP}...")
                chunks = text_splitter.split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                print(f"Creating embeddings using model {EMBEDDING_MODEL_NAME}...")
                embeddings = embedding_model.create_embeddings(chunks, model_name=EMBEDDING_MODEL_NAME)
                collection = vector_store.create_collection(client, collection_name=collection_name)
                print(f"Collection '{collection_name}' created.")
                print("Adding data to Weaviate...")
                vector_store.add_data_to_weaviate(collection, chunks, embeddings)
                print("Data added to Weaviate.")
            else:
                print(f"Collection '{collection_name}' already exists. Indexing skipped.")

        # --- Querying Mode (Modified for Conversation) ---
        elif args.mode == 'query':
            print("Running in QUERY mode (Conversational). Type 'quit' or 'exit' to end.")
            if not client.collections.exists(collection_name):
                print(f"Error: Collection '{collection_name}' does not exist. Please run in 'index' mode first.")
                return

            collection = client.collections.get(collection_name)
            print(f"Using existing collection: '{collection_name}'")

            conversation_history = [] # Initialize conversation history

            while True: # Start conversation loop
                query = input("You: ")
                if query.lower() in ["quit", "exit"]:
                    print("Ending conversation.")
                    break

                conversation_history.append({'role': 'user', 'content': query})

                context = None # Default context to None
                # Conditional RAG based on keyword 'kitiran' (adjust condition as needed)
                if "kitiran" in query.lower():
                    print(f"Query contains 'kitiran', performing RAG...")
                    print(f"Creating query embedding using model {EMBEDDING_MODEL_NAME}...")
                    query_embedding = embedding_model.create_query_embedding(query, model_name=EMBEDDING_MODEL_NAME)

                    print(f"Retrieving top {WEAVIATE_TOP_K} similar chunks...")
                    similar_chunks = vector_store.get_similar_chunks(collection, query_embedding, top_k=WEAVIATE_TOP_K)
                    context = "\n".join([obj.properties["content"] for obj in similar_chunks])
                    print(f"Retrieved context: {context[:200]}...") # Print snippet of context
                else:
                    print("Query does not seem related to 'kitiran', skipping RAG.")

                print("Generating answer...")
                # Pass history and potentially None context to llm
                answer = llm.generate_answer(query=query, context=context, history=conversation_history)

                # Ensure answer is not None before appending (handle potential LLM errors)
                if answer:
                    conversation_history.append({'role': 'assistant', 'content': answer})
                    print(f"Assistant: {answer}")
                else:
                    # If LLM failed, don't add None to history, maybe add error message?
                    print("Assistant: Sorry, I couldn't generate a response.")
                    # Optionally remove the last user message if LLM failed?
                    # conversation_history.pop()


    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if client:
            client.close()
            print("Weaviate connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Application: Index data or Query.")
    parser.add_argument('--mode', type=str, required=True, choices=['index', 'query'],
                        help="Operation mode: 'index' to process and store data, 'query' to ask questions.")
    args = parser.parse_args()
    main(args)