import os
from langchain.document_loaders import PyPDFLoader

# Function to load PDF file content of a given PDF
def load_pdf(relative_file_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, relative_file_path)
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages
