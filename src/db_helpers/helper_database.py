from langchain_community.utilities import SQLDatabase

# Function to get the database schema
def get_schema(db):
    schema = db.get_table_info()
    return schema

# Function to run the SQL query against the database
def run_query(db, query):
    return db.run(query)
