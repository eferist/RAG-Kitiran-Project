import sys
sys.path.append(".")
from src import embedding_model
from src import text_splitter
from src import document_loader
import pytest

def test_create_embeddings():
    text = document_loader.load_document("data/Kitiran Dokumen Context.pdf")
    chunks = text_splitter.split_text(text, chunk_size=150, chunk_overlap=0)
    embeddings = embedding_model.create_embeddings(chunks)
    assert len(embeddings) == len(chunks)
    #assert all(len(embedding) == 768 for embedding in embeddings) # Assuming bert-base-uncased embeddings have a dimension of 768