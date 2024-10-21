from db_helpers.helper_llm import chat_model
from db_helpers.helper_database import get_schema, run_query
from db_helpers.helper_prompt import generate_sql_query, generate_final_answer
from langchain_community.utilities import SQLDatabase

from pdf_helpers.helper_pdf_processing import load_and_process_pdf
from pdf_helpers.helper_llm import create_vectorstore
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

    # Generate SQL query from the user question
    schema = get_schema(db)
    sql_query = generate_sql_query(user_question, schema, chat_history_str)

    # Execute the SQL query and get the response
    sql_response = run_query(db, sql_query)

    # Generate the final natural language answer
    final_answer = generate_final_answer(user_question, sql_response, chat_history_str)

    return sql_query, sql_response, final_answer


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

    # Process the query using invoke()
    response = qa_chain.invoke({"question": user_question, "chat_history": formatted_chat_history})

    answer = response.get('answer', 'No answer generated.')
    source_documents = response.get('source_documents', [])

    # Extract the answer between <ANSWER></ANSWER>
    pattern = r'<ANSWER>\s*(.*?)\s*</ANSWER>'
    match = re.search(pattern, answer, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        # If tags are missing, return the entire answer
        answer = answer.strip()

    return answer, source_documents