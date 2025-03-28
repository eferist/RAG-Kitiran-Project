import sys
sys.path.append(".")
from src import document_loader
from src import text_splitter

def main():
    # Load the document
    text = document_loader.load_document("data/Kitiran Dokumen Context.pdf")

    # Split the text into chunks
    chunks = text_splitter.split_text(text, chunk_size=250, chunk_overlap=10)

    # Print the chunks
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")

if __name__ == "__main__":
    main()