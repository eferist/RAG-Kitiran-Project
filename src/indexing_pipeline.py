# src/indexing_pipeline.py
import time
from typing import List, Dict, Any

# Import the service classes we created
from src.document_processor import DocumentProcessor
from src.llm_service import LLMService
from src.embedding_service import EmbeddingService
from src.vector_db import VectorDB

class IndexingPipeline:
    """
    Orchestrates the process of indexing documents for the RAG system.
    Loads, splits, generates QA pairs, embeds questions, and stores in VectorDB.
    """
    def __init__(self,
                 document_processor: DocumentProcessor,
                 llm_service: LLMService,
                 embedding_service: EmbeddingService,
                 vector_db: VectorDB,
                 collection_name: str,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 0,
                 qa_pairs_per_chunk: int = 3,
                 api_delay_seconds: int = 5):
        """
        Initializes the IndexingPipeline.

        Args:
            document_processor: An instance of DocumentProcessor.
            llm_service: An instance of LLMService.
            embedding_service: An instance of EmbeddingService.
            vector_db: An instance of VectorDB.
            collection_name: The name of the Weaviate collection to use/create.
            chunk_size: Size for splitting documents.
            chunk_overlap: Overlap for splitting documents.
            qa_pairs_per_chunk: Max QA pairs to generate per chunk.
            api_delay_seconds: Delay between LLM calls for QA generation to avoid rate limits.
        """
        self.document_processor = document_processor
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.qa_pairs_per_chunk = qa_pairs_per_chunk
        self.api_delay_seconds = api_delay_seconds

    def run(self, document_path: str, delete_existing: bool = True):
        """
        Executes the indexing pipeline for a given document.

        Args:
            document_path: The path to the document file (e.g., PDF).
            delete_existing: If True, deletes the collection if it already exists before indexing.

        Raises:
            Exception: Propagates exceptions from underlying services.
        """
        print(f"--- Starting Indexing Pipeline for: {document_path} ---")

        # 1. Delete existing collection if requested
        if delete_existing:
            try:
                self.vector_db.delete_collection(self.collection_name)
            except Exception as e:
                print(f"Warning: Failed to delete collection '{self.collection_name}': {e}. Continuing...")
                # Decide if this should be a fatal error

        # 2. Ensure collection exists (Create if not present)
        try:
            self.vector_db.create_collection(self.collection_name)
        except Exception as e:
            print(f"Fatal Error: Could not create or access collection '{self.collection_name}': {e}")
            raise # Stop pipeline if collection cannot be ensured

        # 3. Load and Split Document
        print(f"Processing document: {document_path}")
        try:
            chunks = self.document_processor.load_and_split_pdf(
                file_path=document_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            if not chunks:
                print("No chunks generated from the document. Stopping pipeline.")
                return
            print(f"Document split into {len(chunks)} chunks.")
        except Exception as e:
            print(f"Fatal Error: Failed to load or split document: {e}")
            raise # Stop pipeline if document processing fails

        # 4. Generate QA, Embed Questions, and Prepare Data
        qa_data_list: List[Dict[str, Any]] = []
        total_qa_pairs = 0
        print("Generating QA pairs and embedding questions...")

        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}...")
            try:
                # Generate QA pairs using LLMService
                generated_pairs = self.llm_service.generate_qa_pairs(
                    chunk=chunk,
                    max_pairs=self.qa_pairs_per_chunk
                )

                if not generated_pairs:
                     print(f"    No QA pairs generated for chunk {i+1}.")
                     # Apply delay even if no pairs generated, as API was likely called
                     if self.api_delay_seconds > 0 and i < len(chunks) - 1:
                         print(f"    Pausing for {self.api_delay_seconds} seconds...")
                         time.sleep(self.api_delay_seconds)
                     continue # Move to the next chunk

                print(f"    Generated {len(generated_pairs)} QA pair(s) for chunk {i+1}.")

                # Embed questions and prepare data objects
                for qa_pair in generated_pairs:
                    # Basic validation (already done in LLMService, but double-check doesn't hurt)
                    if "question" not in qa_pair or "answer" not in qa_pair:
                        print(f"    Skipping invalid QA pair format in chunk {i+1}: {qa_pair}")
                        continue

                    try:
                        # Embed the question using EmbeddingService
                        question_embedding = self.embedding_service.embed_query(qa_pair["question"])

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
                         # For now, skipping the pair

                # Apply delay between chunks to avoid hitting API rate limits
                if self.api_delay_seconds > 0 and i < len(chunks) - 1:
                    print(f"    Pausing for {self.api_delay_seconds} seconds before next chunk...")
                    time.sleep(self.api_delay_seconds)

            except Exception as qa_err:
                print(f"    Error processing chunk {i+1} for QA generation: {qa_err}")
                # Decide if we should stop the whole pipeline or just skip the chunk
                # For now, skipping the chunk but logging the error
                # Apply delay even on error if API might have been hit
                if self.api_delay_seconds > 0 and i < len(chunks) - 1:
                    print(f"    Pausing for {self.api_delay_seconds} seconds...")
                    time.sleep(self.api_delay_seconds)
                continue

        print(f"Generated a total of {total_qa_pairs} valid QA pairs with embeddings.")

        # 5. Add data to Weaviate
        if qa_data_list:
            print(f"Adding {len(qa_data_list)} items to Weaviate collection '{self.collection_name}'...")
            try:
                self.vector_db.add_data(self.collection_name, qa_data_list)
                print("Data successfully added to Weaviate.")
            except Exception as e:
                print(f"Fatal Error: Failed to add data to Weaviate: {e}")
                raise # Stop pipeline if data storage fails
        else:
            print("No valid QA data generated or embedded. Nothing added to Weaviate.")

        print(f"--- Indexing Pipeline finished for: {document_path} ---")