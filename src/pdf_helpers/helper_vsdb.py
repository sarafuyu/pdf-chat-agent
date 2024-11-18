from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
from langchain.schema import Document  

import stat
import os

import shutil
from langchain_community.vectorstores import Chroma
from pdf_helpers.helper_llm import embeddings

def load_split_pdf(file_path):
    # Load documents from the PDF file
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0) # Old 500, 100
    docs = text_splitter.split_documents(documents)

    return docs

def save_pdfs(uploaded_files):
    # Save the uploaded PDFs to temporary files
    file_paths = []
    for i, uploaded_file in enumerate(uploaded_files):
        file_path = f"data/pdfs/temp_pdf_{i}.pdf"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

    # Load and process the PDFs
    docs = []
    for file_path in file_paths:
        docs.extend(load_split_pdf(file_path))
    return docs

def load_and_split_pdfs(uploaded_files):
    """
    Load and split PDF documents from uploaded files without saving to disk.
    """
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    # Process each uploaded file directly
    for uploaded_file in uploaded_files:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"  # Extract text from each page

        # Create a Document object compatible with LangChain
        document = Document(page_content=text, metadata={"source": uploaded_file.name})
        split_docs = text_splitter.split_documents([document])  # Split into chunks
        docs.extend(split_docs)  # Add to the docs list

    return docs

def remove_readonly(func, path, excinfo):
    """Error handler for shutil.rmtree to remove read-only files."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def reset_vector_store_db(persist_directory="./data/chroma"):
    """
    Resets the vector store database by removing the persist directory and recreating it.
    """
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory, onerror=remove_readonly)
        except Exception as e:
            print(f"Error removing persist directory: {e}")
    os.makedirs(persist_directory, exist_ok=True)

def create_vectorstore(uploaded_files):
    # Process uploaded files and split into appropriate chunks
    docs = save_pdfs(uploaded_files) # load_and_split_pdfs

    # Create Chroma vector database from documents
    persist_directory = "./data/chroma" # Path to store the Chroma database
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)

    return db

import tempfile

def create_db(uploaded_files):
    """
    Create a Chroma vectorstore database from the uploaded PDF files.

    Args:
        uploaded_files (list): List of uploaded PDF files from Streamlit file_uploader.

    Returns:
        Chroma vectorstore instance.
    """
    # Initialize an empty list to hold all documents
    docs = []

    
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_files.getbuffer())
        tmp_file_path = tmp_file.name

    # Load the PDF document
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    split_docs = text_splitter.split_documents(documents)

    # Add split documents to the list
    docs.extend(split_docs)

    # Remove the temporary file
    os.remove(tmp_file_path)

    # Create embeddings and add to vectorstore (Chroma)
    persist_directory = "./data/chroma"  # Directory to persist the Chroma database

    # Initialize Chroma vectorstore
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vectorstore  

def create_retriever(db):
    # Define retriever for similarity search
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    return retriever