from langchain_community.utilities import SQLDatabase

# Function to get the database schema
def get_schema(db):
    schema = db.get_table_info()
    return schema

# Function to run the SQL query against the database
def run_query(db, query):
    try:
        result = db.run(query)
        return result
    except Exception as e:
        # Handle exceptions and re-raise them to be caught in the calling function
        raise Exception(f"Error executing SQL query: {str(e)}")
