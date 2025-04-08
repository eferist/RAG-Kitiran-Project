# src/document_processor.py
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

class DocumentProcessor:
    """
    Handles loading text content from documents (currently PDFs) and splitting
    it into manageable chunks.
    """

    def __init__(self):
        """Initializes the DocumentProcessor."""
        # Initialization logic can be added here if needed in the future
        pass

    def load_pdf(self, file_path: str) -> str:
        """
        Loads text content from a PDF file.

        Args:
            file_path: The path to the PDF file.

        Returns:
            The extracted text content as a single string.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: For other errors during PDF processing.
        """
        print(f"Loading document from: {file_path}")
        try:
            with pdfplumber.open(file_path) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages)
            print(f"Successfully loaded text from {file_path}.")
            return text
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            raise
        except Exception as e:
            print(f"Error loading or processing PDF file {file_path}: {e}")
            raise RuntimeError(f"Failed to process PDF: {e}") from e

    def split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 0) -> List[str]:
        """
        Splits a given text into chunks using RecursiveCharacterTextSplitter.

        Args:
            text: The text content to split.
            chunk_size: The maximum size of each chunk (in characters).
            chunk_overlap: The number of characters to overlap between chunks.

        Returns:
            A list of text chunks.
        """
        if not text:
            print("Warning: Attempting to split empty text.")
            return []
            
        print(f"Splitting text with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
                # Add other parameters like separators if needed
            )
            chunks = text_splitter.split_text(text)
            print(f"Text split into {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            print(f"Error splitting text: {e}")
            # Depending on requirements, return empty list or raise
            raise RuntimeError(f"Failed to split text: {e}") from e

    def load_and_split_pdf(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 0) -> List[str]:
        """
        Convenience method to load a PDF and split its text in one step.

        Args:
            file_path: The path to the PDF file.
            chunk_size: The maximum size of each chunk (in characters).
            chunk_overlap: The number of characters to overlap between chunks.

        Returns:
            A list of text chunks extracted from the PDF.
        """
        text = self.load_pdf(file_path)
        return self.split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)