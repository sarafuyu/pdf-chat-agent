import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define embedding model using HuggingFace
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"use_auth_token": HUGGINGFACEHUB_API_TOKEN}
)

# Create a HuggingFace endpoint for the LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct", 
    task="text-generation", 
    max_new_tokens=500, 
    do_sample=False, 
    temperature=0.1
)