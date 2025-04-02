# tests/manual_embedding_tester.py
import os
import sys
from dotenv import load_dotenv
from scipy.spatial.distance import cosine
import numpy as np
import time

# --- Path Setup ---
# Add src directory to Python path to allow importing modules from src
tests_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(tests_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# --- Import Custom Modules ---
try:
    from document_loader import load_document
    from text_splitter import split_text
    from embedding_model import create_embeddings, create_query_embedding
except ImportError as e:
    print(f"Error importing modules from src: {e}")
    print("Ensure modules exist in src/ and you are running this script from the project root or that the src path is correct.")
    sys.exit(1)

# --- Configuration ---
# Load environment variables from .env file in the project root
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Parameters for Experimentation ---
# MODIFY THESE VALUES TO TEST DIFFERENT PREPARATION SETTINGS
# Changing these requires re-running the script
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "data/Kitiran Dokumen Context.pdf") # Path relative to project root
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") # Prioritize env var
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" # Force specific model for testing
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30 # Approx 10% of CHUNK_SIZE
TOP_N_RESULTS = 5 # Number of top results to show for each query

# --- Core Logic Functions ---

def prepare_retrieval_data(doc_path, chunk_size, chunk_overlap, model_name):
    """
    Loads, splits, and embeds the document chunks.
    This is the expensive "preparation" step. Run ONCE per script execution.
    """
    print("-" * 60)
    print(f"PREPARING RETRIEVAL DATA...")
    print(f"  Document: '{doc_path}'")
    print(f"  Chunk Size: {chunk_size}, Overlap: {chunk_overlap}")
    print(f"  Embedding Model: {model_name}")
    print("-" * 60)

    prep_start_time = time.time()

    # Ensure document path is absolute or relative to project root
    full_doc_path = doc_path if os.path.isabs(doc_path) else os.path.join(project_root, doc_path)

    if not os.path.exists(full_doc_path):
        print(f"🛑 Error: Document not found at {full_doc_path}")
        return None, None # Indicate failure

    # 1. Load Document
    print("1. Loading document...")
    try:
        document_text = load_document(full_doc_path)
        if not document_text:
            print("🛑 Error: Failed to load document or document is empty.")
            return None, None
        print(f"   Document loaded ({len(document_text)} characters).")
    except Exception as e:
        print(f"🛑 Error loading document: {e}")
        return None, None

    # 2. Split Text
    print(f"2. Splitting text...")
    try:
        chunks = split_text(document_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            print("🛑 Error: Text splitting resulted in no chunks.")
            return None, None
        print(f"   Generated {len(chunks)} chunks.")
    except Exception as e:
        print(f"🛑 Error splitting text: {e}")
        return None, None

    # 3. Create Embeddings for Chunks
    print(f"3. Creating chunk embeddings (this may take a while)...")
    try:
        chunk_embeddings = create_embeddings(chunks, model_name=model_name)
        # Basic validation
        if chunk_embeddings is None or len(chunk_embeddings) != len(chunks):
             print(f"🛑 Error: Embedding creation failed or returned incorrect number of embeddings.")
             return None, None
        # Check if embeddings are valid (simple check for None or empty)
        if any(emb is None or len(emb) == 0 for emb in chunk_embeddings):
             print(f"🛑 Error: Some chunk embeddings are invalid (None or empty).")
             # You might want more sophisticated checks here depending on the model
             return None, None

        print(f"   Chunk embeddings created (shape of first: {np.array(chunk_embeddings[0]).shape if chunk_embeddings else 'N/A'}).")
    except Exception as e:
        print(f"🛑 Error creating chunk embeddings: {e}")
        return None, None

    prep_end_time = time.time()
    print("-" * 60)
    print(f"✅ Data preparation finished in {prep_end_time - prep_start_time:.2f} seconds.")
    print("-" * 60)

    return chunks, chunk_embeddings


def perform_similarity_search(query, query_embedding_func, chunks, chunk_embeddings, model_name, top_n):
    """
    Embeds the query and finds the most similar chunks from the prepared data.
    This is the relatively fast "querying" step.
    """
    if chunk_embeddings is None or chunks is None:
        print("🛑 Error: Cannot perform search, preparation data is missing or invalid.")
        return

    print(f"\n🔍 Searching for query: '{query}'")
    search_start_time = time.time()

    # 1. Embed Query
    try:
        query_embedding = query_embedding_func(query, model_name=model_name)
        if query_embedding is None:
             print("🛑 Error: Failed to create query embedding.")
             return
        # Ensure numpy array
        if not isinstance(query_embedding, np.ndarray): query_embedding = np.array(query_embedding)
        if np.all(query_embedding == 0):
            print("⚠️ Warning: Query embedding resulted in a zero vector. Similarities might be zero.")
            # Continue cautiously or return, depending on desired behavior
    except Exception as e:
        print(f"🛑 Error creating query embedding: {e}")
        return

    # 2. Calculate Cosine Similarities
    similarities = []
    num_chunks = len(chunk_embeddings)
    for i in range(num_chunks):
        chunk_emb = chunk_embeddings[i]
        # Ensure numpy array (should be already, but good practice)
        if not isinstance(chunk_emb, np.ndarray): chunk_emb = np.array(chunk_emb)

        # Basic validation before calculating similarity
        if query_embedding.shape != chunk_emb.shape:
            print(f"⚠️ Warning: Shape mismatch between query ({query_embedding.shape}) and chunk {i} ({chunk_emb.shape}). Skipping chunk {i}.")
            continue
        if np.all(chunk_emb == 0):
             print(f"⚠️ Warning: Zero vector detected for chunk {i}. Skipping chunk {i}.")
             continue
        # Check for NaN/Inf values which can cause issues with cosine distance
        if not np.isfinite(query_embedding).all() or not np.isfinite(chunk_emb).all():
            print(f"⚠️ Warning: Non-finite values (NaN/Inf) detected in query or chunk {i} embedding. Skipping chunk {i}.")
            continue

        try:
            # Cosine Distance = 1 - Cosine Similarity
            distance = cosine(query_embedding, chunk_emb)
            # Handle potential numerical instability giving slightly out-of-range values
            distance = np.clip(distance, 0.0, 2.0) # Cosine distance range
            similarity = 1 - distance
            similarities.append((similarity, i, chunks[i])) # Store score, index, FULL chunk text
        except ValueError as e:
             # This might happen with invalid vectors despite checks
             print(f"🛑 Error calculating cosine similarity for chunk {i}: {e}")
        except Exception as e:
             print(f"🛑 Unexpected error during similarity calculation for chunk {i}: {e}")


    # 3. Sort and Print Results
    similarities.sort(key=lambda x: x[0], reverse=True)

    print(f"\n--- Top {min(top_n, len(similarities))} Results (out of {len(similarities)} calculated) ---")
    if not similarities:
        print("   No similarity results to display.")
    else:
        for rank, (sim, index, chunk_text) in enumerate(similarities[:top_n], 1):
            print(f"Rank {rank}: Score={sim:.4f}, Chunk Index={index}")
            # Display start of chunk - adjust length as needed
            snippet_length = 200
            print(f"  Chunk Start ({len(chunk_text)} chars): '{chunk_text[:snippet_length]}...'")
            print("-" * 10) # Separator between results

    search_end_time = time.time()
    print(f"--- Search finished in {search_end_time - search_start_time:.2f} seconds ---")


# --- Main Execution Logic ---
if __name__ == "__main__":

    # --- 1. Preparation Phase (Runs ONCE when script starts) ---
    # Uses parameters defined at the top of the script
    prepared_chunks, prepared_embeddings = prepare_retrieval_data(
        doc_path=DOCUMENT_PATH,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        model_name=EMBEDDING_MODEL_NAME
    )

    # --- 2. Interactive Query Phase ---
    if prepared_chunks is not None and prepared_embeddings is not None:
        print("\n" + "="*25)
        print("🚀 INTERACTIVE QUERY MODE")
        print("Using prepared data based on script parameters.")
        print("Enter your query below. Type 'quit' or 'exit' to stop.")
        print("="*25 + "\n")

        while True:
            try:
                user_query = input("Enter query: ")
                if user_query.lower() in ['quit', 'exit']:
                    break
                if not user_query: # Handle empty input
                    continue

                # Perform the search using the prepared data
                perform_similarity_search(
                    query=user_query,
                    query_embedding_func=create_query_embedding,
                    chunks=prepared_chunks,              # REUSE prepared data
                    chunk_embeddings=prepared_embeddings, # REUSE prepared data
                    model_name=EMBEDDING_MODEL_NAME,      # Use the same model
                    top_n=TOP_N_RESULTS
                )
            except EOFError: # Handle Ctrl+D
                 print("\nExiting...")
                 break
            except KeyboardInterrupt: # Handle Ctrl+C
                 print("\nExiting...")
                 break
            except Exception as e: # Catch unexpected errors in the loop
                 print(f"\n🛑 An unexpected error occurred in the query loop: {e}")
                 # Optionally decide whether to break or continue
                 # break

    else:
        print("\n🛑 Exiting script because data preparation failed.")

    print("\n👋 Test script finished.")