from transformers import AutoTokenizer, AutoModel
import torch

# Create embeddings for the text chunks
# Removed default value for model_name
def create_embeddings(chunks, model_name: str):
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

# Removed default value for model_name
def create_query_embedding(query, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()