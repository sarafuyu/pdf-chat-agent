import os
from langchain.prompts.prompt import PromptTemplate

# Function to read a text file and store it in a string
def read_file(relative_file_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, relative_file_path)
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content

# Define a custom prompt template with explicit instructions
prompt_template = read_file('prompt_question.txt')
prompt_question = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])
