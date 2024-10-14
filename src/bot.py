import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEndpoint

# Load HuggingFace API key from environment variable
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Function to load and process the PDF
def load_db(file, chain_type="stuff", k=3):
    print(f"Loading PDF: {file}")
    # Load documents
    loader = PyPDFLoader(file)
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Define embedding model using HuggingFace
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"use_auth_token": HUGGINGFACEHUB_API_TOKEN}
    )

    # Create Chroma vector database from documents
    persist_directory = "./data/chroma"  # Path to store the Chroma database
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)

    # Define retriever for similarity search
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Create a conversational retrieval chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-72B-Instruct", 
            task="text-generation", 
            max_new_tokens=150, 
            do_sample=False, 
            temperature=0.1,
        ),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )

    return qa

# Function to handle the conversation loop
def conversation_loop(qa_chain):
    chat_history = []
    
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Exiting the chat.")
            break

        # Process the query with the conversation chain using invoke
        result = qa_chain.invoke({"question": query, "chat_history": chat_history})
        chat_history.extend([(query, result["answer"])])
        
        # Display the concise chatbot's response
        print(f"\nYour question: {query}")
        print(f"Agent answer: {result['answer']}")

        # Show only the highest-ranking source
        print("\n--- Answer Sources ---")
        if result["source_documents"]:
            top_source = result["source_documents"][0]
            print(f"Source: {top_source.metadata['source']}\n{top_source.page_content}\n")

# Function to clear the conversation history
def clear_history():
    return []

# Main function to run the terminal-based chatbot
def main():
    # Load the PDF file into the database
    file_path = input("Enter the path to your PDF file: ")
    qa_chain = load_db(file_path)

    # Start the conversation loop
    print("Chat initialized. Type 'exit' to stop.")
    conversation_loop(qa_chain)

if __name__ == "__main__":
    main()
