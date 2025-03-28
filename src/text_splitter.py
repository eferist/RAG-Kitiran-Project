from langchain.text_splitter import RecursiveCharacterTextSplitter

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