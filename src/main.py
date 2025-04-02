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
            print("Running in INDEX mode (QA Generation).")

            # --- Delete existing collection if it exists ---
            if client.collections.exists(collection_name):
                print(f"Deleting existing collection '{collection_name}'...")
                client.collections.delete(collection_name)
                print(f"Collection '{collection_name}' deleted.")

            # --- Create collection with the new schema ---
            print(f"Creating collection '{collection_name}' with QA schema...")
            collection = vector_store.create_collection(client, collection_name=collection_name)
            print(f"Collection '{collection_name}' created.")

            # --- Load and Chunk Document ---
            print(f"Loading document from {DOCUMENT_PATH}...")
            text = document_loader.load_document(DOCUMENT_PATH)
            print(f"Splitting text with chunk size {CHUNK_SIZE} and overlap {CHUNK_OVERLAP}...")
            chunks = text_splitter.split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            print(f"Found {len(chunks)} chunks.")

            # --- Generate QA, Embed Questions, and Prepare Data ---
            qa_data_list = []
            total_qa_pairs = 0
            print("Generating QA pairs and embedding questions...")
            for i, chunk in enumerate(chunks):
                print(f"  Processing chunk {i+1}/{len(chunks)}...")
                # --- Call LLM to generate QA pairs ---
                # !!! Assumes llm.generate_qa_pairs(chunk) exists and returns a list of {'question': ..., 'answer': ...} dicts
                # !!! We may need to implement/refine this function in src/llm.py
                try:
                    generated_pairs = llm.generate_qa_pairs(chunk) # Placeholder call
                    if not generated_pairs:
                         print(f"    No QA pairs generated for chunk {i+1}.")
                         continue
                    print(f"    Generated {len(generated_pairs)} QA pair(s) for chunk {i+1}.")
                except Exception as qa_err:
                    print(f"    Error generating QA for chunk {i+1}: {qa_err}")
                    continue # Skip this chunk on error

                for qa_pair in generated_pairs:
                    if "question" not in qa_pair or "answer" not in qa_pair:
                        print(f"    Skipping invalid QA pair in chunk {i+1}: {qa_pair}")
                        continue

                    # --- Embed the Question ---
                    try:
                        question_embedding = embedding_model.create_query_embedding(
                            qa_pair["question"], model_name=EMBEDDING_MODEL_NAME
                        )
                        qa_data_list.append({
                            "question": qa_pair["question"],
                            "answer": qa_pair["answer"],
                            "source_chunk": chunk, # Store original chunk
                            "vector": question_embedding
                        })
                        total_qa_pairs += 1
                    except Exception as emb_err:
                         print(f"    Error embedding question for chunk {i+1}: {emb_err}")
                         # Decide whether to skip just this pair or the whole chunk

            print(f"Generated a total of {total_qa_pairs} QA pairs.")

            # --- Add data to Weaviate ---
            if qa_data_list:
                print("Adding QA data to Weaviate...")
                # Use the updated function which expects a list of dicts
                vector_store.add_data_to_weaviate(collection, qa_data_list)
                print("QA data added to Weaviate.")
            else:
                print("No QA data generated or embedded successfully. Nothing added to Weaviate.")

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

                    print(f"Retrieving top {WEAVIATE_TOP_K} similar QA pairs based on question similarity...")
                    # get_similar_chunks returns objects whose *question* vectors matched the query vector
                    similar_objects = vector_store.get_similar_chunks(collection, query_embedding, top_k=WEAVIATE_TOP_K)
                    # Extract the *answer* from the properties of the retrieved objects to use as context
                    context = "\n---\n".join([obj.properties["answer"] for obj in similar_objects if "answer" in obj.properties])
                    # --- Added explicit print for full retrieved context ---
                    print("\n--- Full Retrieved Context ---")
                    print(context)
                    print("--- End Retrieved Context ---\n")
                    # --- End Added Print ---
                    print(f"Retrieved answers for context: {context[:200]}...") # Print snippet of answers
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