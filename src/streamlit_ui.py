import streamlit as st
<<<<<<< Updated upstream
from agent_logic import database_agent, pdf_processing, pdf_agent
import os
=======
from agent_logic import database_agent, pdf_loading, pdf_agent
>>>>>>> Stashed changes

from langchain_community.document_loaders import PyPDFLoader  # Import PyPDFLoader for loading PDFs
from langchain_community.embeddings import HuggingFaceEmbeddings  # Corrected import for embeddings
from langchain_community.vectorstores import FAISS  # Ensure using community version
from langchain.text_splitter import CharacterTextSplitter  # Import text splitter for chunking text
from langchain.chains import ConversationalRetrievalChain  # For conversational queries
from langchain.memory import ConversationBufferMemory  # To keep track of the chat history
from langchain_huggingface import HuggingFaceEndpoint  # Keep this import as is


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
                                      value="How many albums are there in the database?")
        
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
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)

<<<<<<< Updated upstream
        # User input for the question
        user_question = st.text_input("Ask a question about the PDF:", 
                                        value="What is the length of the giraffe's tongue?")

        # Path to the pre-selected PDF file
        default_pdf_path = "data/pdfs/sample.pdf"

        # Check if the user has uploaded a file, otherwise use the default file
        if uploaded_file:
            st.success("PDF uploaded successfully!")
<<<<<<< Updated upstream
=======

            # User input for the question
            user_question = st.text_input("Ask a question about the PDF", 
                                              value="What is the length of the giraffe's tongue?") 
            with st.spinner("Processing PDF..."):
                # Save the uploaded PDF to a temporary file
                i=0
                files = []
                for file in uploaded_file: 
                    i+=1
                    name = "data/pdfs/temp/temp_pdf" + str(i) + ".pdf"
                    files.add(name)
                    
                    with open(name, "wb") as f:
                        f.write(file.getbuffer())
                
                # Call the PDF agent with the uploaded PDF
                if st.button("Run Agent"):
                    with st.spinner("Generating answer..."):
                        answer, source_documents = pdf_agent(files, user_question)

                        st.write("Answer:")
                        st.success(answer)

                        # Show sources if available
                        if source_documents:
                            st.write("Answer Source")
                            top_source = source_documents[0]
                            # st.write(f"Source: {top_source.metadata['source']}")
                            st.code(top_source.page_content)
>>>>>>> Stashed changes
        else:
            st.info(f"No file uploaded, checking default file: {default_pdf_path}")
            # Open the default file if the user doesn't upload one
            if os.path.exists(default_pdf_path):
                uploaded_file = default_pdf_path # open(default_pdf_path, "rb")
                st.success("Using the default PDF file.")
            else:
                st.error(f"Default file not found: {default_pdf_path}")
                st.warning("Please upload a PDF file to proceed.")
                return # Stop execution if neither a file is uploaded nor a default file exists

        with st.spinner("Processing PDF..."):
            qa_chain = pdf_processing(uploaded_file)
            
        # Call the PDF agent with the uploaded PDF
        if st.button("Ask Question"):
            with st.spinner("Generating answer..."):
                answer, source_documents = pdf_agent(qa_chain, user_question)

                st.write("Answer: ")
                st.success(answer)

                # Show sources if available
                if source_documents:
                    st.write("Answer Source: ")
                    top_source = source_documents[0]
                    st.write(f"Source: {top_source.metadata['source']}")
                    st.write(top_source.page_content)
    
=======
        # Conversation chain initialization
        qa_chain = None

        # Load the pdf needed for the agent
        if st.button("Process PDF"):
            
            # Given valid path load vector database
            if uploaded_file:
                
                with st.spinner("Processing PDF..."):

                    # TODO: remove? Save the uploaded PDF to a temporary file
                    with open("data/pdfs/temp_pdf.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Creating the conversation chain connected to the pdf's db 
                    qa_chain = pdf_loading("data/pdfs/temp_pdf.pdf")
                    st.success("PDF uploaded successfully!")
            else:
                st.warning("Please upload a valid PDF file to proceed.")
>>>>>>> Stashed changes

            # User input for the question
            user_question = st.text_input("Ask a question about the PDF:", 
                                        value="What is the length of the giraffe's tongue?")
    
            # Call the PDF agent with the uploaded PDF
            if st.button("Ask Question"):
                
                with st.spinner("Generating answer..."):
                    answer, source_documents = pdf_agent(qa_chain, user_question)

                    st.write("Answer:")
                    st.success(answer)

                    # Show sources if available
                    if source_documents:
                        st.write("\n--- Answer Sources ---")
                        top_source = source_documents[0]
                        st.write(f"Source: {top_source.metadata['source']}")
                        st.write(top_source.page_content)
                
        
if __name__ == '__main__':
    main()
