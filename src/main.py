from helpers.helper_llm import chat_model
from helpers.helper_database import db, run_query, get_schema
from helpers.helper_prompt import generate_sql_query, generate_final_answer

def main():
    user_question = 'how many albums are there in the database?'

    # Generate the SQL query from the user's question
    schema = get_schema()
    sql_query = generate_sql_query(user_question, schema)
    print("Generated SQL Query:")
    print(sql_query)

    # Execute the SQL query and get the response
    sql_response = run_query(sql_query)
    print("\nSQL Response:")
    print(sql_response)

    # Generate the final natural language answer
    final_answer = generate_final_answer(user_question, sql_query, sql_response)
    print("\nFinal Answer:")
    print(final_answer)

if __name__ == "__main__":
    main()
