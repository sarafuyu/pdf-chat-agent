from langchain_community.utilities import SQLDatabase

# Load the database
mysql_uri = 'mysql+mysqlconnector://root:Kaka1234!!@localhost:3306/chinook'  # Update with your MySQL credentials
db = SQLDatabase.from_uri(mysql_uri)

# Function to get the database schema
def get_schema():
    schema = db.get_table_info()
    return schema

# Function to run the SQL query against the database
def run_query(query):
    return db.run(query)
