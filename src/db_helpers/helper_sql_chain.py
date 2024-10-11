import re
from helper_llm import invoke_llm

# Function to generate SQL query from user's question
def generate_sql_query(db, question, prompt_formatter):
    schema = db.get_table_info()
    prompt = prompt_formatter(schema=schema, question=question)
    sql_query = invoke_llm(prompt)

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
