from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts.prompt import PromptTemplate

def create_qa_chain(retriever, chain_type="stuff"):

    # Create a HuggingFace endpoint for the LLM
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct", 
        task="text-generation", 
        max_new_tokens=500, 
        do_sample=False, 
        temperature=0.1
    )

    # Define a custom prompt template with explicit instructions
    prompt_template = """You are a helpful assistant that provides concise answers to the user's questions based on the following context.

Context:
{context}

Conversation History:
{chat_history}

Question:
{question}

Provide a concise and direct natural language answer to the user's question, using the context above as needed.

Your answer should be in complete sentences and should not include any additional text or conversation.

Provide only the final answer enclosed in <ANSWER></ANSWER> tags.

For example:

<ANSWER>Your answer here.</ANSWER>

(Note: Do not include any additional text outside of the <ANSWER></ANSWER> tags.)

Answer:
"""
    # Note: Removed the <ANSWER> tag at the end of the prompt and added an example.

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question", "chat_history"]
    )

    # Create the conversational retrieval chain with the custom prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_llm=llm,
        chain_type=chain_type,
        verbose=False,
        combine_docs_chain_kwargs={'prompt': prompt},
        return_source_documents=True,
    )

    return qa_chain