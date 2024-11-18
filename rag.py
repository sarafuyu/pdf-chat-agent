import os

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter


# Get the Hugging Face Hub API token from environment variable
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define model using HuggingFace for vectorization
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"use_auth_token": HUGGINGFACEHUB_API_TOKEN}
)

# Create a HuggingFace endpoint for the LLM used for answer generation
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=500,
    do_sample=False,
    temperature=0.1
)

# Path to your PDF file
pdf_path = 'src/data/pdfs/Animal_facts.pdf'  # Replace with your PDF file path

# Load the PDF document
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create embeddings and add to vectorstore (Chroma)
persist_directory = "./src/data/chroma"  # Directory to persist the Chroma database

# Initialize Chroma vectorstore
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

# Create a retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Perform a search and print the results
query = "Enter your query here"  # Replace with your query
retrieved_docs = retriever.invoke(query)

# Print the results
for i, doc in enumerate(retrieved_docs):
    print(f"Result {i+1}:\n{doc.page_content}\n")
