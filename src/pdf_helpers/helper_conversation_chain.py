from langchain.chains import ConversationalRetrievalChain

from pdf_helpers.helper_llm import llm
from pdf_helpers.helper_prompt import prompt_question

def create_qa_chain(retriever, chain_type="stuff"):

    # Create the conversational retrieval chain with the custom prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_llm=llm,
        chain_type=chain_type,
        verbose=False,
        combine_docs_chain_kwargs={'prompt': prompt_question},
        return_source_documents=True,
    )

    return qa_chain
