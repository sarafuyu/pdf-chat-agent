import os
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEndpoint

# Set up Hugging Face Hub API token
os.getenv("HUGGINGFACEHUB_API_TOKEN")

def create_qa_chain(retriever, chain_type="stuff"):

    # Create a conversational retrieval chain
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct", 
        task="text-generation", 
        max_new_tokens=150, 
        do_sample=False, 
        temperature=0.1
    )

    # Create a chatbot chain. Memory is managed externally.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True
    )

    return qa_chain
