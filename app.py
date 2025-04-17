import streamlit as st
import asyncio
from main import results  # This uses your main.py's logic

st.set_page_config(page_title="Jarvis AI Assistant", layout="wide")
st.title("🤖 Jarvis AI Assistant")

# For storing chat history in Streamlit session
if "history" not in st.session_state:
    st.session_state.history = []

# Input
user_input = st.text_input("You:", key="input")

# Process Input
if user_input:
    st.session_state.history.append(("user", user_input))

    # Run async logic from main.py
    response = asyncio.run(results(user_input))

    if isinstance(response, str):
        st.session_state.history.append(("bot", response))
    elif isinstance(response, bytes):  # image
        st.session_state.history.append(("bot", "[image]"))
        st.image(response, caption="Generated Image")
    else:
        st.session_state.history.append(("bot", str(response)))

# Display chat history
for sender, message in st.session_state.history:
    with st.chat_message(sender):
        st.markdown(message)
