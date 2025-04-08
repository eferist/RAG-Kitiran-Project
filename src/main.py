import sys
import argparse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path if running as script
# (Consider using proper packaging instead for larger projects)
sys.path.append(os.path.dirname(__file__))

# Import the new service and pipeline classes
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from llm_service import LLMService
from vector_db import VectorDB
from indexing_pipeline import IndexingPipeline
from query_pipeline import QueryPipeline

# --- Load Configuration from Environment Variables ---
# Consider moving this to a dedicated config module/class later
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "data/Kitiran Dokumen Context.pdf") # Default to provided PDF
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 0))
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT", "8080")
WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "KitiranDocsQA") # More specific name
WEAVIATE_TOP_K = int(os.getenv("WEAVIATE_TOP_K", 5))
QA_PAIRS_PER_CHUNK = int(os.getenv("QA_PAIRS_PER_CHUNK", 3))
API_DELAY_SECONDS = int(os.getenv("API_DELAY_SECONDS", 5)) # Delay for LLM calls in indexing
# --- End Configuration Loading ---

def run_indexing():
    """Initializes services and runs the indexing pipeline."""
    print("--- Initializing Services for Indexing ---")
    vector_db = None # Initialize to None for finally block
    try:
        doc_processor = DocumentProcessor()
        llm_service = LLMService() # Assumes API key is handled internally
        embedding_service = EmbeddingService() # Assumes API key is handled internally
        vector_db = VectorDB(host=WEAVIATE_HOST, port=WEAVIATE_PORT)

        indexing_pipeline = IndexingPipeline(
            document_processor=doc_processor,
            llm_service=llm_service,
            embedding_service=embedding_service,
            vector_db=vector_db,
            collection_name=WEAVIATE_COLLECTION_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            qa_pairs_per_chunk=QA_PAIRS_PER_CHUNK,
            api_delay_seconds=API_DELAY_SECONDS
        )

        print("\n--- Starting Indexing ---")
        # Run the pipeline, optionally deleting existing data
        indexing_pipeline.run(document_path=DOCUMENT_PATH, delete_existing=True)
        print("--- Indexing Complete ---")

    except (ValueError, ConnectionError, RuntimeError) as e:
        print(f"ERROR during indexing initialization or execution: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during indexing: {e}")
    finally:
        if vector_db:
            vector_db.close()

def run_query():
    """Initializes services and runs the query pipeline in a loop."""
    print("--- Initializing Services for Querying ---")
    vector_db = None # Initialize to None for finally block
    try:
        llm_service = LLMService()
        embedding_service = EmbeddingService()
        vector_db = VectorDB(host=WEAVIATE_HOST, port=WEAVIATE_PORT)

        # Check if collection exists before starting query loop
        if not vector_db.collection_exists(WEAVIATE_COLLECTION_NAME):
             print(f"ERROR: Collection '{WEAVIATE_COLLECTION_NAME}' does not exist.")
             print("Please run the script in 'index' mode first.")
             return # Exit if collection is missing

        query_pipeline = QueryPipeline(
            llm_service=llm_service,
            embedding_service=embedding_service,
            vector_db=vector_db,
            collection_name=WEAVIATE_COLLECTION_NAME,
            top_k=WEAVIATE_TOP_K
        )

        print("\n--- Starting Conversational Query Mode ---")
        print("Type 'quit' or 'exit' to end.")
        conversation_history = []

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit"]:
                    print("Ending conversation.")
                    break

                # Add user message to history *before* processing
                conversation_history.append({'role': 'user', 'content': user_input})

                # Process query using the pipeline
                assistant_response = query_pipeline.process_query(
                    query=user_input,
                    history=conversation_history # Pass current history
                )

                print(f"Assistant: {assistant_response}")

                # Add assistant response to history
                conversation_history.append({'role': 'assistant', 'content': assistant_response})

                # Optional: Limit history size
                # if len(conversation_history) > MAX_HISTORY_LENGTH * 2:
                #     conversation_history = conversation_history[-MAX_HISTORY_LENGTH*2:]

            except (ValueError, ConnectionError, RuntimeError) as e:
                 print(f"ERROR during query processing: {e}")
                 # Optionally remove last user message from history on error?
                 # conversation_history.pop()
            except EOFError: # Handle Ctrl+D or unexpected end of input
                 print("\nInput stream closed. Ending conversation.")
                 break
            except KeyboardInterrupt: # Handle Ctrl+C
                 print("\nInterrupted by user. Ending conversation.")
                 break


    except (ValueError, ConnectionError, RuntimeError) as e:
        print(f"ERROR during query service initialization: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during query mode: {e}")
    finally:
        if vector_db:
            vector_db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Application: Index data or Query.")
    parser.add_argument('--mode', type=str, required=True, choices=['index', 'query'],
                        help="Operation mode: 'index' to process and store data, 'query' to ask questions.")
    # Add other arguments if needed (e.g., --document-path, --collection-name)
    args = parser.parse_args()

    if args.mode == 'index':
        run_indexing()
    elif args.mode == 'query':
        run_query()
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        parser.print_help()