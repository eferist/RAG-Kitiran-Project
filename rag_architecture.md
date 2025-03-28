# RAG Architecture for Customer Support Questions

## Overview

This document describes the architecture for a Retrieval-Augmented Generation (RAG) program that answers customer support questions based on product documentation.

## Architecture Diagram

```mermaid
graph LR
    A[Product Documentation (PDF)] --> B(Document Loader)
    B --> C(Text Splitter)
    C --> D(Embedding Model)
    D --> E(Vector Store)
    F[User Query] --> G(Embedding Model)
    G --> H(Similarity Search)
    H --> E
    E --> I(Context Retrieval)
    I --> J(LLM)
    J --> K[Answer]
```

## Components

*   **Document Loader:** Loads the PDF document.
*   **Text Splitter:** Splits the document into smaller chunks of text.
*   **Embedding Model:** Creates vector embeddings of the text chunks.
*   **Vector Store:** Stores the vector embeddings for efficient similarity search.
*   **User Query:** The question asked by the user.
*   **Embedding Model:** Creates a vector embedding of the user query.
*   **Similarity Search:** Finds the text chunks in the vector store that are most similar to the user query.
*   **Context Retrieval:** Retrieves the original text chunks from the document based on the similarity search results.
*   **LLM (Large Language Model):** Generates an answer to the user query based on the retrieved context.
*   **Answer:** The final answer presented to the user.

## Data Flow

1.  The product documentation (PDF) is loaded by the Document Loader.
2.  The Document Loader splits the document into smaller chunks of text using the Text Splitter.
3.  The Embedding Model creates vector embeddings of the text chunks.
4.  The vector embeddings are stored in the Vector Store.
5.  When a user asks a question (User Query), the Embedding Model creates a vector embedding of the query.
6.  The Similarity Search finds the text chunks in the Vector Store that are most similar to the query embedding.
7.  The Context Retrieval retrieves the original text chunks from the document.
8.  The LLM uses the retrieved context to generate an answer to the user query.
9.  The Answer is presented to the user.