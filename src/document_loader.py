import pdfplumber

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