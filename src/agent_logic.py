from db_helpers.helper_llm import chat_model
from db_helpers.helper_database import get_schema, run_query
from db_helpers.helper_prompt import generate_sql_query, generate_final_answer
from langchain_community.utilities import SQLDatabase

from pdf_helpers.helper_pdf_processing import load_and_process_pdf
from pdf_helpers.helper_vector_db import create_vectorstore
from pdf_helpers.helper_conversation_chain import create_qa_chain

# Database chat agent logic
def database_agent(mysql_uri=None, user_question=None):
    if not mysql_uri:
        mysql_uri = 'mysql+mysqlconnector://root:Kaka1234!!@localhost:3306/chinook'  # Default MySQL URI
    db = SQLDatabase.from_uri(mysql_uri)
    
    if not user_question:
        user_question = 'How many albums are there in the database?'  # Default question

    # Generate SQL query from the user question
    schema = get_schema(db)
    sql_query = generate_sql_query(user_question, schema)

    # Execute the SQL query and get the response
    sql_response = run_query(db, sql_query)

    # Generate the final natural language answer
    final_answer = generate_final_answer(user_question, sql_query, sql_response)
    
    return sql_query, sql_response, final_answer


# PDF chat agent logic
<<<<<<< Updated upstream
def pdf_processing(file_path=None):
=======

def pdf_loading(file_path):
>>>>>>> Stashed changes
    if not file_path:
        file_path = "C:/Users/esaydrr/OneDrive - Ericsson/Desktop/dna-projects/pdf-chat-agent/src/data/pdfs/Animal_facts.pdf"  # Default PDF

    # Load and process the PDF
    docs = load_and_process_pdf(file_path)
    
    # Create the vector store and retriever
    retriever = create_vectorstore(docs)
    
    # Create the conversational chain
    qa_chain = create_qa_chain(retriever)

    return qa_chain

def pdf_agent(qa_chain, user_question=None):
    
    
    return qa_chain

def pdf_agent(qa_chain, user_question):
    
    # Process the query
    response = qa_chain.invoke({"question": user_question, "chat_history": []})
    answer = response.get('answer', 'No answer generated.')
    
    return answer, response.get("source_documents", [])