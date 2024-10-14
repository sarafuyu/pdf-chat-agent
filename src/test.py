import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.vectorstores import DocArrayInMemorySearch

# Set up Hugging Face Hub API token
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load PDF and extract text
def load_pdf(file_path):
    # load documents
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def doc_splitting(docs):
    # Create an instance of RecursiveCharacterTextSplitter
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=0,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )

    # Split the documents into chunks
    split_docs = r_splitter.split_documents(docs)  # Pass the list of documents directly

    return split_docs

def create_vector_db(split_docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"use_auth_token": HUGGINGFACEHUB_API_TOKEN}
    )
    vectorstore = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings, 
        persist_directory="data/chroma"
    )
    return vectorstore

def load_db(file, chain_type, k):
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"use_auth_token": HUGGINGFACEHUB_API_TOKEN}
    )
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm= HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct", task="text-generation", max_new_tokens=150, do_sample=False, temperature=0.1,), # ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 

def main():
    # Load and process the PDF file
    pdf_path = input("Enter the path to your PDF file: ")
    docs = load_pdf(pdf_path)  # Load the PDF file
    split_docs = doc_splitting(docs)  # Split the documents
    vectorstore = create_vector_db(split_docs) # Create the vector store

    # Define the LLM (Hugging Face Endpoint)
    # llm = HuggingFaceEndpoint(repo_id="gpt2", model_kwargs={"temperature": 0.2})
    # Initialize the Hugging Face Endpoint with a model suitable for instruction following
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        task="text-generation",
        max_new_tokens=150,
        do_sample=False,
        temperature=0.1,
    )

    # Build prompt
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(), return_source_documents=True, chain_type="refine", chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    
    # Define memory for conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

if __name__ == "__main__":
    main()
