#from db_helpers.helper_llm import chat_model
from db_helpers.helper_database import get_schema, run_query
from db_helpers.helper_prompt import generate_sql_query, generate_final_answer
from langchain_community.utilities import SQLDatabase
import traceback

# from pdf_helpers.helper_pdf_processing import load_and_process_pdf
# from pdf_helpers.helper_vsdb import create_vectorstore
from pdf_helpers.helper_conversation_chain import create_qa_chain
import re

# Database chat agent logic
def database_agent(mysql_uri=None, user_question=None, chat_history=[]):
    if not mysql_uri:
        mysql_uri = 'mysql+mysqlconnector://root:Kaka1234!!@localhost:3306/chinook'  # Default MySQL URI
    db = SQLDatabase.from_uri(mysql_uri)

    if not user_question:
        user_question = 'How many albums are there in the database?'  # Default question

    # Format chat history as a string
    chat_history_str = '\n'.join(chat_history)

    try:
        # Generate SQL query from the user question
        schema = get_schema(db)
        sql_query = generate_sql_query(user_question, schema, chat_history_str)

        # Execute the SQL query and get the response
        sql_response = run_query(db, sql_query)

        # Check if the SQL response is empty
        if not sql_response:
            final_answer = "I'm sorry, I couldn't find any results for your query."
        else:
            # Generate the final natural language answer
            final_answer = generate_final_answer(user_question, sql_response, chat_history_str)

        return sql_query, sql_response, final_answer

    except Exception as e:
        # Handle exceptions and return an error message
        error_message = f"An error occurred: {str(e)}"
        print(traceback.format_exc())  # Optionally log the traceback
        return None, None, error_message


# PDF chat agent logic
def pdf_agent(retriever, user_question, chat_history=[]):
    # Create the conversational chain
    qa_chain = create_qa_chain(retriever)

    # Format chat history as a list of tuples (user, assistant)
    formatted_chat_history = []
    for i in range(0, len(chat_history), 2):
        user_msg = chat_history[i].replace("User: ", "")
        assistant_msg = chat_history[i+1].replace("Assistant: ", "") if i+1 < len(chat_history) else ""
        formatted_chat_history.append((user_msg, assistant_msg))

    # Process the query 
    response = qa_chain.invoke({"question": user_question, "chat_history": formatted_chat_history})

    answer = response.get('answer', 'No answer generated.')
    source_documents = response.get('source_documents', [])
    top_source = source_documents[0].page_content

    # Extract the answer between <ANSWER></ANSWER>
    pattern = r'<ANSWER>\s*(.*?)\s*</ANSWER>'
    match = re.search(pattern, answer, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        # If tags are missing, return the entire answer
        answer = answer.strip()

    return answer, top_source