import os
from langchain_huggingface import HuggingFaceEmbeddings

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define embedding model using HuggingFace
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"use_auth_token": HUGGINGFACEHUB_API_TOKEN}
)
