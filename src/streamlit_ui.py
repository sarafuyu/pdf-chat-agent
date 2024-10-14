import streamlit as st
from agent_logic import database_agent, pdf_agent

from langchain_community.document_loaders import PyPDFLoader  # Import PyPDFLoader for loading PDFs
from langchain_community.embeddings import HuggingFaceEmbeddings  # Corrected import for embeddings
from langchain_community.vectorstores import FAISS  # Ensure using community version
from langchain.text_splitter import CharacterTextSplitter  # Import text splitter for chunking text
from langchain.chains import ConversationalRetrievalChain  # For conversational queries
from langchain.memory import ConversationBufferMemory  # To keep track of the chat history
from langchain_huggingface import HuggingFaceEndpoint  # Keep this import as is

import tempfile
from PyPDF2 import PdfReader

def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Converting text to chunks
def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)   
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Using all-MiniLM embeddings model and Faiss to get vectorstore
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Creating custom template to guide LLM model
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

from langchain.prompts import PromptTemplate

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

from dotenv import load_dotenv
import os

load_dotenv()

# Custom function to create conversation chain with HuggingFace model
def get_conversationchain(vectorstore):
    # Load a HuggingFace model
    llm = HuggingFaceEndpoint(repo_id="gpt2", temperature=0.2)  # Set temperature directly here

    # Create a memory buffer to store chat history
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True,
                                      output_key='answer')
    
    # Create the conversation chain using the HuggingFace model and vectorstore retriever
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory
    )
    
    return conversation_chain

def process_pdf_content(pdf_text):
    """Process the PDF text and prepare it for querying."""
    # Implement text chunking and embedding logic
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(pdf_text)
    
    # Create embeddings and a vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

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
        uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"], accept_multiple_files=True)

        if uploaded_file:
            st.success("PDF uploaded successfully!")

            # Button to process the query
            if st.button("Process PDF"):
                # Get the PDF text
                raw_text = get_pdf_text(uploaded_file)
                
                # Get the text chunks
                text_chunks = get_chunks(raw_text)
                
                # Create vectorstore
                vectorstore = get_vectorstore(text_chunks)
                
                # Create conversation chain
                st.session_state.conversation = get_conversationchain(vectorstore)

                # Default question
                user_question = st.text_input("Ask a question about the PDF file:", 
                                            value="Can you give a short summary of the PDF content?")
                
            # Button to process the question
            if st.button("Ask Question"):
                with st.spinner("Generating answer..."):
                    # Generate the answer using the conversation chain
                    response = st.session_state.conversation({'question': user_question})
                    
                    # Extract the answer from the response
                    answer = response.get('answer', 'No answer generated.')
                    
                    st.write("Answer:")
                    st.success(answer)

        else:
            st.warning("Please upload a PDF file to proceed.")

if __name__ == '__main__':
    main()
