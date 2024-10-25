from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from pdf_helpers.helper_llm import embeddings

def load_split_pdf(file_path):
    # Load documents from the PDF file
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
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

def create_vectorstore(uploaded_files):
    # Process uploaded files and split into appropriate chunks
    docs = save_pdfs(uploaded_files)

    # Create Chroma vector database from documents
    persist_directory = "./data/chroma" # Path to store the Chroma database
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)

    return db

def create_retriever(db):

    # Define retriever for similarity search
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    return retriever