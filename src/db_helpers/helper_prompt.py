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

    # Extract the SQL query between <SQL_QUERY> and </SQL_QUERY>
    pattern = r'<SQL_QUERY>\s*(.*?)\s*</SQL_QUERY>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        sql_query = match.group(1).strip()
    else:
        # If tags are missing, extract code blocks or return the response as is
        pattern_code = r"```(?:sql)?\s*(.*?)\s*```"
        match_code = re.search(pattern_code, response, re.DOTALL | re.IGNORECASE)
        if match_code:
            sql_query = match_code.group(1).strip()
        else:
            sql_query = response.strip()

    # Ensure the SQL query ends with a semicolon
    if not sql_query.endswith(';'):
        sql_query += ';'

    return sql_query

# Create the prompt template for generating the final answer
template_response = read_file('prompt_answer.txt')
prompt_response = PromptTemplate(
    input_variables=["chat_history", "question", "response"],
    template=template_response
)

# Function to generate the final natural language answer
def generate_final_answer(question, response, chat_history):
    prompt = prompt_response.format(
        chat_history=chat_history, question=question, response=response)
    final_response = chat_model.invoke(prompt)

    # Extract the answer between <ANSWER> and </ANSWER>
    pattern = r'<ANSWER>\s*(.*?)\s*</ANSWER>'
    match = re.search(pattern, final_response, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        # If tags are missing, return the entire response
        answer = final_response.strip()
    return answer
