# src/vector_db.py
import weaviate
import weaviate.classes as wvc
from weaviate.exceptions import UnexpectedStatusCodeError
from typing import List, Dict, Any

class VectorDB:
    """
    Provides an abstraction layer for interacting with the Weaviate vector database.
    Manages client connection, collection creation/deletion, data insertion, and querying.
    """
    def __init__(self, host: str = "http://localhost", port: str = "8080"):
        """
        Initializes the VectorDB service and connects to the Weaviate instance.

        Args:
            host: The hostname or IP address of the Weaviate instance.
            port: The port number of the Weaviate instance.

        Raises:
            ConnectionError: If connection to Weaviate fails or the client is not ready.
        """
        self.host = host
        self.port = port
        self.client: weaviate.WeaviateClient | None = None
        try:
            print(f"Attempting connection to Weaviate at {self.host}:{self.port}")
            self.client = weaviate.connect_to_local(
                host=self.host,
                port=int(self.port) # Ensure port is integer
            )
            # Check connection status
            if not self.client.is_ready():
                 raise ConnectionError("Weaviate client connected but not ready.")
            print("Successfully connected to Weaviate.")
        except Exception as e:
            print(f"Failed to connect to Weaviate at {self.host}:{self.port}: {e}")
            # Raise a specific error to be handled upstream
            raise ConnectionError(f"Could not connect to Weaviate: {e}") from e

    def close(self):
        """Closes the connection to Weaviate."""
        if self.client:
            self.client.close()
            print("Weaviate connection closed.")
            self.client = None

    def _ensure_connected(self):
        """Internal helper to check for an active client connection."""
        if not self.client:
            raise ConnectionError("Weaviate client is not connected or has been closed.")

    def collection_exists(self, collection_name: str) -> bool:
        """Checks if a collection exists."""
        self._ensure_connected()
        return self.client.collections.exists(collection_name)

    def create_collection(self, collection_name: str):
        """
        Creates a new collection with a predefined schema for QA data.
        Does nothing if the collection already exists.

        Args:
            collection_name: The name for the new collection.

        Returns:
            The Weaviate collection object (either newly created or existing).

        Raises:
            RuntimeError: If collection creation fails unexpectedly.
            ConnectionError: If the client is not connected.
        """
        self._ensure_connected()
        try:
            if self.client.collections.exists(collection_name):
                print(f"Collection '{collection_name}' already exists. Using existing collection.")
                return self.client.collections.get(collection_name)
            else:
                print(f"Creating collection '{collection_name}'...")
                collection = self.client.collections.create(
                    name=collection_name,
                    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                    properties=[
                        wvc.config.Property(name="question", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="answer", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="source_chunk", data_type=wvc.config.DataType.TEXT)
                    ]
                )
                print(f"Collection '{collection_name}' created successfully.")
                return collection
        except Exception as e:
            print(f"Failed to create or get collection '{collection_name}': {e}")
            raise RuntimeError(f"Could not create or get collection: {e}") from e

    def delete_collection(self, collection_name: str):
        """Deletes a collection if it exists."""
        self._ensure_connected()
        try:
            if self.client.collections.exists(collection_name):
                print(f"Deleting collection '{collection_name}'...")
                self.client.collections.delete(collection_name)
                print(f"Collection '{collection_name}' deleted successfully.")
            else:
                print(f"Collection '{collection_name}' does not exist, skipping deletion.")
        except Exception as e:
            print(f"Failed to delete collection '{collection_name}': {e}")
            raise RuntimeError(f"Could not delete collection: {e}") from e

    def get_collection(self, collection_name: str):
        """Gets an existing collection object."""
        self._ensure_connected()
        if not self.client.collections.exists(collection_name):
             raise ValueError(f"Collection '{collection_name}' does not exist.")
        try:
            return self.client.collections.get(collection_name)
        except Exception as e:
            print(f"Failed to get collection '{collection_name}': {e}")
            raise RuntimeError(f"Could not get collection: {e}") from e


    def add_data(self, collection_name: str, qa_data_list: List[Dict[str, Any]]):
        """
        Adds QA data to the specified Weaviate collection.

        Args:
            collection_name: The name of the collection to add data to.
            qa_data_list: A list of dictionaries, where each dictionary must have
                          'question', 'answer', 'source_chunk', and 'vector' keys.

        Raises:
            ValueError: If the collection does not exist or data format is incorrect.
            RuntimeError: If data insertion fails.
            ConnectionError: If the client is not connected.
        """
        self._ensure_connected()
        try:
            collection = self.get_collection(collection_name) # Reuse get_collection logic
            objects_to_insert = []
            for i, qa_data in enumerate(qa_data_list):
                # Validate required keys
                required_keys = {"question", "answer", "source_chunk", "vector"}
                if not required_keys.issubset(qa_data):
                    raise ValueError(f"Missing required keys in qa_data item at index {i}: {required_keys - set(qa_data)}")

                obj = wvc.data.DataObject(
                    properties={
                        "question": qa_data["question"],
                        "answer": qa_data["answer"],
                        "source_chunk": qa_data["source_chunk"]
                    },
                    vector=qa_data["vector"] # Vector corresponds to the question
                )
                objects_to_insert.append(obj)

            if objects_to_insert:
                print(f"Inserting {len(objects_to_insert)} objects into '{collection_name}'...")
                collection.data.insert_many(objects_to_insert)
                print("Data insertion successful.")
            else:
                print("No valid data objects provided to insert.")

        except ValueError as ve: # Re-raise specific validation errors
             raise ve
        except Exception as e:
            print(f"Failed to add data to collection '{collection_name}': {e}")
            raise RuntimeError(f"Could not add data: {e}") from e

    def search_similar(self, collection_name: str, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Performs a similarity search in the specified collection.

        Args:
            collection_name: The name of the collection to search in.
            query_embedding: The embedding vector of the query.
            top_k: The maximum number of similar items to retrieve.

        Returns:
            A list of dictionaries, where each dictionary represents a retrieved object
            containing its properties and potentially metadata like distance/score.

        Raises:
            ValueError: If the collection does not exist.
            RuntimeError: If the search query fails.
            ConnectionError: If the client is not connected.
        """
        self._ensure_connected()
        try:
            collection = self.get_collection(collection_name) # Reuse get_collection logic
            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=top_k
            )
            # Return properties of the found objects
            # Adjust based on what the calling code needs (e.g., include vector, distance?)
            return [obj.properties for obj in response.objects]
        except ValueError as ve: # Re-raise specific validation errors
             raise ve
        except Exception as e:
            print(f"Failed to search in collection '{collection_name}': {e}")
            raise RuntimeError(f"Could not perform search: {e}") from e