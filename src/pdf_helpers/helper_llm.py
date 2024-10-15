import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def create_vectorstore(docs):

    # Define embedding model using HuggingFace
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"use_auth_token": HUGGINGFACEHUB_API_TOKEN}
    )

    # Create Chroma vector database from documents
    persist_directory = "./data/chroma" # Path to store the Chroma database
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)

    # Define retriever for similarity search
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    return retriever
