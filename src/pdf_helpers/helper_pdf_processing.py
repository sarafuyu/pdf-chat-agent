import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_process_pdf(file):

    # Load documents
    loader = PyPDFLoader(file)
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    return docs