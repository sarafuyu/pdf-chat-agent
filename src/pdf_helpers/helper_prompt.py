def conversation_loop(qa_chain):
    chat_history = []

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Exiting the chat.")
            break

        # Process the query with the conversation chain using invoke
        result = qa_chain.invoke({"question": query, "chat_history": chat_history})
        chat_history.extend([(query, result["answer"])])

        # Display the concise chatbot's response
        print(f"\nYour question: {query}")
        print(f"Agent answer: {result['answer']}")

        # Show only the highest-ranking source
        if result["source_documents"]:
            top_source = result["source_documents"][0]
            print(f"Source: {top_source.metadata['source']}\n{top_source.page_content}\n")

def clear_history():
    return []
