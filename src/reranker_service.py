# src/reranker_service.py
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, Dict, Any

class RerankerService:
    """
    Handles the reranking of retrieved documents using a Cross-Encoder model.
    """
    def __init__(self, model_name: str = 'BAAI/bge-reranker-base', device: str = 'cpu'):
        """
        Initializes the RerankerService and loads the Cross-Encoder model.

        Args:
            model_name: The name of the Cross-Encoder model on Hugging Face Hub.
            device: The device to run the model on ('cpu', 'cuda').
        """
        print(f"Loading reranker model: {model_name}...")
        try:
            # Use sentence-transformers CrossEncoder for convenience
            self.model = CrossEncoder(model_name, max_length=512, device=device) # Adjust max_length if needed
            print(f"Reranker model '{model_name}' loaded successfully on {device}.")
        except Exception as e:
            print(f"Error loading reranker model '{model_name}': {e}")
            raise

    def rerank(self, query: str, documents_props: List[Dict[str, Any]], text_key: str = "answer", top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on their relevance to a query.

        Args:
            query: The search query string.
            documents_props: A list of dictionaries, where each dictionary represents
                             a retrieved document and its properties (must contain
                             the key specified by `text_key`).
            text_key: The key in the document dictionaries that contains the text to be reranked.
            top_n: The maximum number of documents to return after reranking.

        Returns:
            A sorted list of the top_n document dictionaries, ordered by relevance score (highest first).
            Returns an empty list if input documents_props is empty or text_key is missing.
        """
        if not documents_props:
            print("Reranker received no documents to rerank.")
            return []

        # Prepare pairs for the CrossEncoder: [ [query, passage_text], [query, passage_text], ... ]
        sentence_pairs = []
        valid_docs_with_indices = [] # Keep track of original index and props for docs that have the text_key

        for i, props in enumerate(documents_props):
            if text_key in props and props[text_key]: # Ensure the key exists and text is not empty
                sentence_pairs.append([query, props[text_key]])
                valid_docs_with_indices.append({'index': i, 'props': props})
            else:
                print(f"Warning: Document at index {i} missing '{text_key}' or has empty content. Skipping.")

        if not sentence_pairs:
            print(f"Reranker found no documents with valid text under key '{text_key}'.")
            return []

        print(f"Reranking {len(sentence_pairs)} documents for query: '{query[:50]}...'")
        try:
            # Compute scores
            scores = self.model.predict(sentence_pairs, show_progress_bar=False) # Add show_progress_bar=True for long lists
        except Exception as e:
            print(f"Error during reranker prediction: {e}")
            # Decide how to handle: return empty, return original top N, or raise?
            # For robustness, let's return empty here, signaling failure to rerank.
            return []

        # Combine scores with the original valid document properties
        results = []
        for i, score in enumerate(scores):
            original_doc_info = valid_docs_with_indices[i]
            # Add the score to the properties dictionary for potential later use
            original_doc_info['props']['rerank_score'] = float(score)
            results.append(original_doc_info['props']) # Append the whole props dict

        # Sort results by score in descending order
        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Return the top N results
        top_results = results[:top_n]
        print(f"Reranking complete. Selected top {len(top_results)} documents.")
        if top_results:
            print(f"Top reranked score: {top_results[0]['rerank_score']:.4f}")

        return top_results