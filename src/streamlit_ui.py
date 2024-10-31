import streamlit as st
from agent_logic import database_agent, pdf_agent, load_css
from pdf_helpers.helper_vsdb import reset_vector_store_db, create_vectorstore, create_retriever
from ui_helpers.helper_chat_history import display_chat_history
import os

def main():
    st.set_page_config(page_title="Database & PDF Chat Agents", page_icon=":books:")
    st.header("Chat with Database or PDF")

    load_css("ui_helpers/style.css")

    # Initialize conversation histories and retriever in session state
    if "db_chat_history" not in st.session_state:
        st.session_state.db_chat_history = []
    if "pdf_chat_history" not in st.session_state:
        st.session_state.pdf_chat_history = []
        reset_vector_store_db() # Reset the vector store database
    if "pdf_retriever" not in st.session_state:
        st.session_state.pdf_retriever = None

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

        if mysql_uri and user_question: 

            # Button to process the query
            if st.button("Run Agent"):
                with st.spinner("Generating answer..."):
                    # Build the conversation history
                    chat_history = st.session_state.db_chat_history.copy()
                    # Append the current user question
                    chat_history.append(f"User: {user_question}")

                    # Pass the conversation history to the agent
                    sql_query, sql_response, final_answer = database_agent(
                        mysql_uri, user_question, chat_history)

                    # Update chat history with assistant's answer
                    st.session_state.db_chat_history.append(f"User: {user_question}")
                    st.session_state.db_chat_history.append(f"Assistant: {final_answer}")

                    # Display results
                    st.write("##### Generated SQL Query:")
                    st.code(sql_query)

                    st.write("##### SQL Response:")
                    st.code(sql_response)

                    st.write("##### Answer:")
                    st.success(final_answer)

        # Display conversation history only if there are messages
        if st.session_state.db_chat_history:
            st.write("##### Conversation History:")
            display_chat_history(st.session_state.db_chat_history)

    elif agent_choice == "PDF Agent":
        st.subheader("PDF Agent")

        # File uploader for the PDFs (allows multiple files)
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_files:
            # Button to process the files
            if st.button("Process Files"):
                with st.spinner("Processing PDFs..."):
                    # Create the vector store and database retriever
                    st.session_state.pdf_db = create_vectorstore(uploaded_files)
                    st.session_state.pdf_retriever = create_retriever(st.session_state.pdf_db)

        # User question field for PDF questions
        user_question = st.text_input("Ask a question about the PDF(s)",
                                      value="How long is the tongue of a giraffe?")

        # If the retriever is available, display the question input and run agent
        if st.session_state.pdf_retriever and user_question:
            if st.button("Run Agent"):
                with st.spinner("Generating answer..."):
                    # Build the conversation history
                    chat_history = st.session_state.pdf_chat_history.copy()
                    # Append the current user question
                    chat_history.append(f"User: {user_question}")

                    # Call the PDF agent with the retriever and chat history
                    answer, source_document = pdf_agent(
                        st.session_state.pdf_retriever, user_question, chat_history)

                    # Update chat history with assistant's answer
                    st.session_state.pdf_chat_history.append(f"User: {user_question}")
                    st.session_state.pdf_chat_history.append(f"Assistant: {answer}")

                    # Show source document content
                    st.write("##### Source Document:")
                    st.markdown(f'''
                        <div class="source-document">
                            {source_document}
                        </div>
                    ''', unsafe_allow_html=True)

                    # Display results
                    st.write("##### Answer:")
                    st.success(answer)

        # Display conversation history only if there are messages
        if st.session_state.pdf_chat_history:
            st.write("##### Conversation History:")
            display_chat_history(st.session_state.pdf_chat_history)

if __name__ == '__main__':
    main()
