from pdf_helpers.helper_pdf_processing import load_and_process_pdf
from pdf_helpers.helper_llm import create_vectorstore
from pdf_helpers.helper_conversation_chain import create_qa_chain
from pdf_helpers.helper_prompt import conversation_loop

def main():

    # C:/Users/esaydrr/OneDrive - Ericsson/Desktop/dna-projects/pdf-chat-agent/src/data/pdfs/Animal_facts.pdf
    file_path = input("Enter the path to your PDF file: ") 
    
    # Load and process the PDF
    docs = load_and_process_pdf(file_path)
    
    # Create the vector store and retriever
    retriever = create_vectorstore(docs)
    
    # Create the conversational chain
    qa_chain = create_qa_chain(retriever)
    
    print("Chat initialized. Type 'exit' to stop.")
    
    # Start the conversation loop
    conversation_loop(qa_chain)

if __name__ == "__main__":
    main()
