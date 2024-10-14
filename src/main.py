import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint

# Load environment variables from .env file
load_dotenv()

# Define constants
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
CHROMA_DB_PATH = "../data/chroma"

# Load PDF and extract text
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Create embeddings and a vector store
def create_vectorstore(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"use_auth_token": HUGGINGFACE_API_TOKEN})
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=CHROMA_DB_PATH)
    return vectorstore

# Define the main function
def main():
    # Load and process the PDF file
    pdf_path = input("Enter the path to your PDF file: ")
    documents = load_pdf(pdf_path)

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create the vector store
    vectorstore = create_vectorstore(texts)

    # Define memory for conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Define the LLM (Hugging Face Endpoint)
    llm = HuggingFaceEndpoint(repo_id="gpt2", model_kwargs={"temperature": 0.2})

    # Build prompt
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(), chain_type="refine")

    # Start querying the PDF
    while True:
        question = input("\nAsk a question about the PDF (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        result = qa_chain({"query": question})
        print(f"Answer: {result['result']}\n")

if __name__ == "__main__":
    main()
