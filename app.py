import streamlit as st
from main import results, Username  # Assuming main.py has results() and Username

# Initialize session_state for chat logs if not already initialized
if "chat_logs" not in st.session_state:
    st.session_state.chat_logs = []

# Streamlit page config
st.set_page_config(page_title="TaskMind - AI Assistant", page_icon="🤖")
st.title("🧠 TaskMind AI Assistant")
st.markdown("Ask me anything! I understand tasks, general queries, and real-time questions.")

# Input box
user_input = st.text_input("📝 Enter your query:")

# Button to submit
if st.button("Submit") and user_input.strip():
    st.markdown("⏳ Processing your request...")
    
    # Get response from your AI assistant
    response = results(Username, user_input)
    
    # Store the user input and response in session_state chat logs
    st.session_state.chat_logs.append({
        "user": user_input,
        "assistant": response
    })
    
    # Display the response
    st.markdown("### 💬 Response:")
    st.write(response)

# Optionally display the conversation history for this session
if st.session_state.chat_logs:
    st.markdown("### 🗨️ Chat History:")
    for entry in st.session_state.chat_logs:
        st.write(f"**User:** {entry['user']}")
        st.write(f"**Assistant:** {entry['assistant']}")
