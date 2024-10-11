from db_helpers.helper_llm import chat_model
from db_helpers.helper_database import get_schema, run_query
from db_helpers.helper_prompt import generate_sql_query, generate_final_answer
from langchain_community.utilities import SQLDatabase

# Database chat agent logic
def database_agent(mysql_uri=None, user_question=None):
    if not mysql_uri:
        mysql_uri = 'mysql+mysqlconnector://root:Kaka1234!!@localhost:3306/chinook'  # Default MySQL URI
    db = SQLDatabase.from_uri(mysql_uri)
    
    if not user_question:
        user_question = 'how many albums are there in the database?'  # Default question

    # Generate SQL query from the user question
    schema = get_schema(db)
    sql_query = generate_sql_query(user_question, schema)

    # Execute the SQL query and get the response
    sql_response = run_query(db, sql_query)

    # Generate the final natural language answer
    final_answer = generate_final_answer(user_question, sql_query, sql_response)
    
    return sql_query, sql_response, final_answer


# Placeholder for PDF agent logic
def pdf_agent():
    # Implement PDF-related functionality here
    return "PDF agent not yet implemented"
