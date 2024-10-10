import re
from langchain.prompts import PromptTemplate
from helpers.helper_llm import chat_model

# Create the prompt template for generating SQL queries
template_sql = """You are a SQL expert. Given the following database schema, generate a SQL query to answer the user's question.

Schema:
{schema}

Question: {question}

Please provide only the SQL query, without any additional text or explanations."""

prompt_sql = PromptTemplate(input_variables=["schema", "question"], template=template_sql)

# Function to generate SQL query from user's question
def generate_sql_query(question, schema):
    prompt = prompt_sql.format(schema=schema, question=question)
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
template_response = """Given the user's question, the SQL query, and the SQL response, provide a concise natural language answer.

Question: {question}

SQL Query: {query}

SQL Response: {response}

Answer:"""

prompt_response = PromptTemplate(
    input_variables=["question", "query", "response"],
    template=template_response
)

# Function to generate the final natural language answer
def generate_final_answer(question, query, response):
    prompt = prompt_response.format(
        question=question, query=query, response=response)
    final_response = chat_model.invoke(prompt)
    answer = final_response.strip()
    return answer
