import weaviate
import weaviate.classes as wvc
import numpy as np

def connect_to_local():
    client = weaviate.connect_to_local()
    return client

def create_collection(collection_name="Question"):
    client = connect_to_local()
    collection = client.collections.create(
        name=collection_name,
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        properties=[
            wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT)
        ]
    )
    return collection

def add_data_to_weaviate(collection, chunks, embeddings):
    objects_to_insert = []
    for i, chunk in enumerate(chunks):
        obj = wvc.data.DataObject(
            properties={"content": chunk},
            vector=embeddings[i]
        )
        objects_to_insert.append(obj)
    collection.data.insert_many(objects_to_insert)

def get_similar_chunks(collection, query_embedding, top_k=5):
    response = collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k
    )
    return response.objects
