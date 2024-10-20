import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_process_pdf(files):

    documents = []
    # Load documents
    for file in files: 
        loader = PyPDFLoader(file)
        doc = loader.load()
        documents.add(doc)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)

    docs = []
    for doc in documents:
        split = text_splitter.split_documents(doc)
        docs.add(split)


    return split