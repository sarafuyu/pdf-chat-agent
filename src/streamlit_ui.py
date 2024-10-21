import streamlit as st
from agent_logic import database_agent, pdf_agent
from pdf_helpers.helper_pdf_processing import load_and_process_pdf
from pdf_helpers.helper_llm import create_vectorstore


def main():
    st.set_page_config(page_title="Database & PDF Chat Agents", page_icon=":books:")
    st.header("Chat with Database or PDF")

    # Initialize conversation histories and retriever in session state
    if "db_chat_history" not in st.session_state:
        st.session_state.db_chat_history = []
    if "pdf_chat_history" not in st.session_state:
        st.session_state.pdf_chat_history = []
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

        # Button to process the query
        if st.button("Run Agent"):
            with st.spinner("Running agent..."):
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
                st.write("Generated SQL Query:")
                st.code(sql_query)

                st.write("SQL Response:")
                st.code(sql_response)

                st.write("Final Answer:")
                st.success(final_answer)

                # Display conversation history
                st.write("Conversation History:")
                for chat in st.session_state.db_chat_history:
                    st.write(chat)

    elif agent_choice == "PDF Agent":
        st.subheader("PDF Agent")

        # File uploader for the PDFs (allows multiple files)
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!")

            # Button to process the files
            if st.button("Process Files"):
                with st.spinner("Processing PDFs..."):
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
                        docs.extend(load_and_process_pdf(file_path))

                    # Create the vector store and retriever
                    st.session_state.pdf_retriever = create_vectorstore(docs)

                    st.success("PDFs processed successfully! You can now ask questions.")

        else:
            st.warning("Please upload PDF files to proceed.")

        # If the retriever is available, display the question input and run agent
        if st.session_state.pdf_retriever is not None:
            user_question = st.text_input("Ask a question about the PDFs")

            if st.button("Run Agent"):
                with st.spinner("Generating answer..."):
                    # Build the conversation history
                    chat_history = st.session_state.pdf_chat_history.copy()
                    chat_history.append(f"User: {user_question}")

                    # Call the PDF agent with the retriever and chat history
                    answer, source_documents = pdf_agent(
                        st.session_state.pdf_retriever, user_question, chat_history)

                    # Update chat history with assistant's answer
                    st.session_state.pdf_chat_history.append(f"User: {user_question}")
                    st.session_state.pdf_chat_history.append(f"Assistant: {answer}")

                    st.write("Answer:")
                    st.success(answer)

                    # Show source document content
                    if source_documents:
                        st.write("Source Document:")
                        top_source = source_documents[0]
                        st.write(top_source.page_content)

                    # Display conversation history
                    st.write("Conversation History:")
                    for chat in st.session_state.pdf_chat_history:
                        st.write(chat)
    else:
        st.warning("Please upload PDF files to proceed.")

if __name__ == '__main__':
    main()
