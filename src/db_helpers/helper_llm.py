import os
from langchain_huggingface import HuggingFaceEndpoint

# Set up Hugging Face Hub API token
os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the Hugging Face Endpoint with a model suitable for instruction following
chat_model = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=150,
    do_sample=False,
    temperature=0.1,
)
