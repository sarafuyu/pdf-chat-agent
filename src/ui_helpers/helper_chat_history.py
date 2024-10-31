import streamlit as st

# Define a reusable function to display chat history
def display_chat_history(chat_history):
    if chat_history:  # Display only if there are messages
        for chat in chat_history:
            if chat.strip():  # Skip empty or whitespace-only entries
                if chat.startswith("User:"):
                    message = chat.replace("User:", "").strip()
                    st.markdown(f'''
                        <div class="chat-message user">
                            <p>{message}</p>
                        </div>
                    ''', unsafe_allow_html=True)
                elif chat.startswith("Assistant:"):
                    message = chat.replace("Assistant:", "").strip()
                    st.markdown(f'''
                        <div class="chat-message assistant">
                            <p>{message}</p>
                        </div>
                    ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)