from helpers.helper_llm import chat_model
from helpers.helper_database import get_schema, run_query
from helpers.helper_prompt import generate_sql_query, generate_final_answer

from langchain_community.utilities import SQLDatabase

def database_agent():
    
    # Load the specified MySQL database
    mysql_uri = input("What is the database connection to your database? For using the default one press enter: ")
    if not mysql_uri:
        mysql_uri = 'mysql+mysqlconnector://root:Kaka1234!!@localhost:3306/chinook'  # Update with your MySQL credentials
    db = SQLDatabase.from_uri(mysql_uri)
    
    # Get the user question for the database
    user_question = input("Write your question here, for using the default question press enter: ")
    if not user_question:
        user_question = 'how many albums are there in the database?'

    # Generate the SQL query from the user's question
    schema = get_schema(db)

    sql_query = generate_sql_query(user_question, schema)
    print("\nGenerated SQL Query:")
    print(sql_query)

    # Execute the SQL query and get the response
    sql_response = run_query(db, sql_query)
    print("\nSQL Response:")
    print(sql_response)

    # Generate the final natural language answer
    final_answer = generate_final_answer(user_question, sql_query, sql_response)
    print("\nFinal Answer:")
    print(final_answer)

def pdf_agent():
    # Too be implemented, for now i want a menue tab to choose either to do database or pdf questions
    print("Hello")

if __name__ == "__main__":

    while True:
        agent = input("For using the database chat agent type 1, for using the pdf chat agent type 2, press enter to end application: ")
        
        if agent == "1":
            database_agent()
        elif agent == "2":
            pdf_agent()
        elif agent == "":
            print("Closing down appication")
            break
        else:
            print("That was not one of the options. Please type either 1 or 2 to continue")