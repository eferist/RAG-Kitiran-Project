import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch

# Load the document
def load_document(path):
    # Open the PDF file
    with pdfplumber.open(path) as pdf:
        # Initialize an empty string to store the text
        text = ""
        # Iterate over each page in the PDF
        for page in pdf.pages:
            # Extract the text from the page and append it to the string
            text += page.extract_text()
    # Return the extracted text
    return text

# Split the text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=0):
    # Create a RecursiveCharacterTextSplitter object
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    # Return the chunks
    return chunks

# Create embeddings for the text chunks
def create_embeddings(chunks, model_name="bert-base-uncased"):
    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Define a function to get the embedding for a single text
    def get_embedding(text):
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # Disable gradient calculation
        with torch.no_grad():
            # Get the model outputs
            outputs = model(**inputs)
        # Return the mean of the last hidden state
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Create embeddings for all chunks
    embeddings = [get_embedding(chunk) for chunk in chunks]
    # Return the embeddings
    return embeddings