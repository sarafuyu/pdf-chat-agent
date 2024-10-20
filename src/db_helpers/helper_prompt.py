import os
import re
from langchain.prompts import PromptTemplate
from db_helpers.helper_llm import chat_model

# Function to read a text file and store it in a string
def read_file(relative_file_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, relative_file_path)
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content

# Create the prompt template for generating SQL queries
template_sql = read_file('prompt_question.txt')
prompt_sql = PromptTemplate(input_variables=["chat_history", "schema", "question"], template=template_sql)

# Function to generate SQL query from user's question
def generate_sql_query(question, schema, chat_history):
    prompt = prompt_sql.format(chat_history=chat_history, schema=schema, question=question)
    response = chat_model.invoke(prompt)
    sql_query = response.strip()

    # Regular expression pattern to match code blocks
    pattern = r"```(?:sql)?\s*(.*?)\s*```"

    # Search for the pattern in the response
    match = re.search(pattern, sql_query, re.DOTALL | re.IGNORECASE)
    if match:
        sql_query = match.group(1).strip()
    else:
        # If no code block, remove leading/trailing backticks and whitespace
        sql_query = sql_query.strip('`').strip()

    # Ensure the SQL query ends with a semicolon
    if not sql_query.endswith(';'):
        sql_query += ';'

    return sql_query

# Create the prompt template for generating the final answer
template_response = read_file('prompt_answer.txt')
prompt_response = PromptTemplate(
    input_variables=["chat_history", "question", "query", "response"],
    template=template_response
)

# Function to generate the final natural language answer
def generate_final_answer(question, query, response, chat_history):
    prompt = prompt_response.format(
        chat_history=chat_history, question=question, query=query, response=response)
    final_response = chat_model.invoke(prompt)
    answer = final_response.strip()
    return answer
