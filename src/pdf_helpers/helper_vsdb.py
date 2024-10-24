from langchain_community.vectorstores import Chroma
from pdf_helpers.helper_llm import embeddings

def create_vectorstore(docs):
    # Create Chroma vector database from documents
    persist_directory = "./data/chroma" # Path to store the Chroma database
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)

    # Define retriever for similarity search
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    return retriever