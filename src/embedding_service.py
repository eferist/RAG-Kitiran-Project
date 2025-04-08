# src/embedding_service.py
import os
from dotenv import load_dotenv
from google import generativeai as genai

# Load environment variables (consider moving to a central config loader later)
load_dotenv()

class EmbeddingService:
    """
    Encapsulates interactions with the embedding model (Google Gemini).
    Handles API configuration and embedding generation for documents and queries.
    """
    # Define the Gemini embedding model to use
    EMBEDDING_MODEL_NAME = "models/embedding-001"

    def __init__(self):
        """
        Initializes the EmbeddingService, configuring the Google Generative AI client.
        """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. EmbeddingService cannot operate.")
        
        # Configure the client upon initialization
        # Note: genai.configure is global. If other services also call it, ensure consistency.
        try:
            genai.configure(api_key=self.api_key)
            print("Google Generative AI configured for EmbeddingService.")
        except Exception as e:
            # Handle potential configuration errors
            raise RuntimeError(f"Failed to configure Google Generative AI: {e}") from e

    def embed_documents(self, chunks: list[str]) -> list[list[float]]:
        """
        Creates embeddings for a list of text chunks (documents).

        Args:
            chunks: A list of strings, where each string is a document chunk.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
            Returns an empty list if the input is empty.
        """
        if not chunks:
            return []
        
        # Note: The API might have limits on the number of chunks per request.
        # Consider adding batching logic here if necessary for large inputs.
        try:
            result = genai.embed_content(
                model=self.EMBEDDING_MODEL_NAME,
                content=chunks,
                task_type="RETRIEVAL_DOCUMENT" # Task type for embedding documents/chunks
            )
            # The API returns a list of lists of floats when input is a list
            return result['embedding']
        except Exception as e:
            print(f"Error embedding documents: {e}")
            # Depending on requirements, you might return empty list, None, or re-raise
            raise RuntimeError(f"Failed to embed documents: {e}") from e


    def embed_query(self, query: str) -> list[float]:
        """
        Creates an embedding for a single query string.

        Args:
            query: The query string to embed.

        Returns:
            The embedding as a list of floats.
        """
        if not query:
            # Handle empty query case if necessary, maybe return None or raise error
            raise ValueError("Cannot embed an empty query.")
            
        try:
            result = genai.embed_content(
                model=self.EMBEDDING_MODEL_NAME,
                content=query,
                task_type="RETRIEVAL_QUERY" # Task type for embedding search queries
            )
            # The embedding is directly returned as a list of floats for single content
            return result['embedding']
        except Exception as e:
            print(f"Error embedding query: {e}")
            # Depending on requirements, you might return None or re-raise
            raise RuntimeError(f"Failed to embed query: {e}") from e