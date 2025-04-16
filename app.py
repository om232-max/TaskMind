import streamlit as st
from main import results, Username
import os

if "chat_logs" not in st.session_state:
    st.session_state.chat_logs = []

st.set_page_config(page_title="TaskMind - AI Assistant", page_icon="🤖")
st.title("🧠 TaskMind AI Assistant")
st.markdown("Ask me anything! I understand tasks, general queries, and real-time questions.")

user_input = st.text_input("📝 Enter your query:")

if st.button("Submit") and user_input.strip():
    st.markdown("⏳ Processing your request...")
    
    response = results(Username, user_input)
    
    st.session_state.chat_logs.append({
        "user": user_input,
        "assistant": response
    })
    
    st.markdown("### 💬 Response:")
    if response.strip().startswith("![Generated Image]"):
        image_path = response.split("(")[-1].rstrip(")")
        if os.path.exists(image_path):
            st.image(image_path)
            with open(image_path, "rb") as file:
                st.download_button(
                    label="⬇️ Download Image",
                    data=file,
                    file_name=os.path.basename(image_path),
                    mime="image/jpeg"
                )
    else:
        st.write(response)

if st.session_state.chat_logs:
    st.markdown("### 🗨️ Chat History:")
    for entry in st.session_state.chat_logs:
        st.write(f"**User:** {entry['user']}")
        if entry["assistant"].strip().startswith("![Generated Image]"):
            image_path = entry["assistant"].split("(")[-1].rstrip(")")
            if os.path.exists(image_path):
                st.image(image_path)
                with open(image_path, "rb") as file:
                    st.download_button(
                        label=f"⬇️ Download Image ({os.path.basename(image_path)})",
                        data=file,
                        file_name=os.path.basename(image_path),
                        mime="image/jpeg"
                    )
        else:
            st.write(f"**Assistant:** {entry['assistant']}")
