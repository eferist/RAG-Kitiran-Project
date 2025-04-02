import weaviate
import weaviate.classes as wvc
import numpy as np

# Modified to accept host and port
def connect_to_local(host="http://localhost", port="8080"):
    print(f"Attempting connection to Weaviate at {host}:{port}") # Added print for clarity
    client = weaviate.connect_to_local(
        host=host,
        port=int(port) # Ensure port is integer
    )
    return client

# Modified to accept client and collection_name (no default)
def create_collection(client: weaviate.WeaviateClient, collection_name: str):
    # Removed internal connect_to_local() call
    collection = client.collections.create(
        name=collection_name,
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        properties=[
            wvc.config.Property(name="question", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="answer", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="source_chunk", data_type=wvc.config.DataType.TEXT) # Added source_chunk
        ]
    )
    return collection

# Modified to accept a list of QA data objects
def add_data_to_weaviate(collection, qa_data_list):
    """
    Adds QA data to the Weaviate collection.

    Args:
        collection: The Weaviate collection object.
        qa_data_list: A list of dictionaries, where each dictionary has
                      'question', 'answer', 'source_chunk', and 'vector' keys.
    """
    objects_to_insert = []
    for qa_data in qa_data_list:
        obj = wvc.data.DataObject(
            properties={
                "question": qa_data["question"],
                "answer": qa_data["answer"],
                "source_chunk": qa_data["source_chunk"]
            },
            vector=qa_data["vector"] # Vector corresponds to the question
        )
        objects_to_insert.append(obj)
    collection.data.insert_many(objects_to_insert)

# Modified to accept top_k (no default)
def get_similar_chunks(collection, query_embedding, top_k: int):
    response = collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k # Use passed top_k
    )
    return response.objects
