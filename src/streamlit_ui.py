# streamlit_ui.py

import streamlit as st
from agent_logic import database_agent, pdf_agent

from langchain_community.document_loaders import PyPDFLoader  # Import PyPDFLoader for loading PDFs
from langchain.embeddings import HuggingFaceEmbeddings  # Import embeddings for vector storage
from langchain.vectorstores import faiss  # Import Faiss for vector storage
from langchain.text_splitter import CharacterTextSplitter  # Import text splitter for chunking text
from langchain.chains import ConversationalRetrievalChain  # For conversational queries
from langchain.memory import ConversationBufferMemory  # To keep track of the chat history

def load_pdf(uploaded_file):
    """Load PDF and return text content."""
    loader = PyPDFLoader(uploaded_file)  # Use PyPDFLoader to load the PDF
    documents = loader.load()  # Load the documents from the PDF
    text = ""
    for doc in documents:
        text += doc.page_content  # Concatenate text from each page
    return text

def process_pdf_content(pdf_text):
    """Process the PDF text and prepare it for querying."""
    # Here you can implement text chunking and embedding logic
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(pdf_text)
    
    # Create embeddings and a vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vectorstore

def main():
    st.set_page_config(page_title="Database & PDF Chat Agents", page_icon=":books:")
    st.header("Chat with Database or PDF")
    
    # Selection for either the database agent or the PDF agent
    agent_choice = st.sidebar.radio("Select Agent", ("Database Agent", "PDF Agent"))

    if agent_choice == "Database Agent":
        st.subheader("Database Agent")
        
        # Input for the MySQL connection URI
        mysql_uri = st.text_input("Enter your MySQL connection URI", 
                                  value='mysql+mysqlconnector://root:Kaka1234!!@localhost:3306/chinook')
        
        # Input for the user's database query
        user_question = st.text_input("Ask a question about the database", 
                                      value="how many albums are there in the database?")
        
        # Button to process the query
        if st.button("Run Query"):
            with st.spinner("Running query..."):
                sql_query, sql_response, final_answer = database_agent(mysql_uri, user_question)
                
                # Display results
                st.write("Generated SQL Query:")
                st.code(sql_query)
                
                st.write("SQL Response:")
                st.code(sql_response)
                
                st.write("Final Answer:")
                st.success(final_answer)

    elif agent_choice == "PDF Agent":
        st.subheader("PDF Agent")
        
        # File uploader for the PDF
        uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"], accept_multiple_files=False)

        if uploaded_file:
            st.success("PDF uploaded successfully!")
            pdf_text = load_pdf(uploaded_file)  # Load text from PDF

            # Process the content from the PDF
            vectorstore = process_pdf_content(pdf_text)
            st.session_state.vectorstore = vectorstore  # Store vectorstore in session state for retrieval

            # Default question
            user_question = st.text_input("Ask a question about the PDF file:", 
                                          value="Can you give a short summary of the PDF content?")
            
            # Button to process the question
            if st.button("Ask Question"):
                with st.spinner("Generating answer..."):
                    # Call your logic to get the answer based on the processed PDF content
                    response = "This is where the response will be generated based on the PDF content."
                    
                    # Replace the line above with the actual response generation logic using vectorstore and a chain
                    st.write("Answer:")
                    st.success(response)

        else:
            st.warning("Please upload a PDF file to proceed.")

if __name__ == '__main__':
    main()
