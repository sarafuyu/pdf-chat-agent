# Database and PDF Chat Agent

This project implements two interactive agents:

1. **Database Agent**: Allows users to interact with a SQL database using natural language queries. The agent translates these queries into SQL commands, executes them, and returns the results in natural language.

2. **PDF Agent**: Enables users to ask questions about the content of one or more PDF documents. The agent processes the PDFs, creates a searchable vector database, and provides answers based on the content.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Database Agent](#database-agent)
  - [PDF Agent](#pdf-agent)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

---

## Features

### Database Agent

- Translates natural language questions into SQL queries.
- Executes queries against a MySQL database.
- Provides answers in natural language.
- Maintains conversation history for context-aware interactions.
- Handles errors gracefully and provides meaningful feedback.

### PDF Agent

- Processes one or more PDF documents.
- Creates a vector database for efficient document retrieval.
- Answers user questions based on the content of the PDFs.
- Provides source excerpts related to the answers.
- Maintains conversation history for context-aware interactions.

---

## Installation

### Prerequisites

- Python 3.7 or higher
- MySQL server (for the Database Agent)
- Virtual environment tool (optional but recommended)

### Steps

1. **Clone the Repository**

   ```bash
   git clone ...
   cd ...
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**

   - Obtain a Hugging Face API token.
   - Create a `.env` file in the project root directory and add:

     ```env
     HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here
     ```

---

## Usage

Run the Streamlit application:

```bash
streamlit run src/streamlit_ui.py
```

This will open the application in your default web browser.

### Database Agent

1. **Select "Database Agent"** from the sidebar.

2. **Enter MySQL Connection URI**

   - Default:

     ```
     mysql+mysqlconnector://root:password@localhost:3306/chinook
     ```

   - Modify according to your database credentials.

3. **Ask a Question**

   - Input a natural language question about your database.

4. **Run Agent**

   - Click the "Run Agent" button.
   - View the generated SQL query, SQL response, and the final answer.

5. **Conversation History**

   - Interact with the agent using follow-up questions.
   - The agent uses conversation history for context.

### PDF Agent

1. **Select "PDF Agent"** from the sidebar.

2. **Upload PDF Files**

   - Use the file uploader to select one or more PDF documents.

3. **Process Files**

   - Click the "Process Files" button.
   - The agent will process the PDFs and create a vector database.

4. **Ask a Question**

   - Input a natural language question about the content of the PDFs.

5. **Run Agent**

   - Click the "Run Agent" button.
   - View the answer and related source document content.

6. **Conversation History**

   - Continue asking questions.
   - The agent maintains context using conversation history.

---

## Project Structure

```
chat-agent/
├── src/
│   ├── data/
│   │   ├── database/
│   │   │   └── Chinook_MySql.sql
│   │   └── pdfs/
│   │       ├── Animal_facts.docx
│   │       └── General_info_etc_2024.pdf
│   ├── db_helpers/
│   │   ├── helper_database.py
│   │   ├── helper_llm.py
│   │   ├── helper_prompt.py
│   │   ├── prompt_answer.txt
│   │   └── prompt_question.txt
│   ├── pdf_helpers/
│   │   ├── helper_conversation_chain.py
│   │   ├── helper_llm.py
│   │   ├── helper_prompt.py
│   │   ├── helper_vsdb.py
│   │   └── prompt_question.txt
│   ├── agent_logic.py
│   ├── example_questions.txt
│   └── streamlit_ui.py
├── .gitignore
├── Databricks to Snowflake Conversion.docx
├── RAG Framework evaluation.docx
├── README.md
├── SQL Agent Framework.docx
├── Zephyr and Mistral Notes.docx
└── requirements.txt
```

---

## Dependencies

- **Streamlit**: Web application framework for the UI.
- **LangChain**: Framework for building applications with LLMs.
- **HuggingFace Hub**: Access to pre-trained language models.
- **SQLAlchemy**: Database toolkit for SQL in Python.
- **PyPDFLoader**: For loading PDF documents.
- **Chroma**: Vector database for document retrieval.
- **Other Libraries**: `mysql-connector-python`, `dotenv`, `re`, `os`, etc.

*(Refer to `requirements.txt` for the full list.)*

---

## Acknowledgements

- **OpenAI**: For language model APIs and guidance.
- **Hugging Face**: For providing access to a wide range of models.
- **LangChain**: For simplifying LLM application development.