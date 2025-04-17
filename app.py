import streamlit as st
from main import results, Username
import os

st.set_page_config(page_title="TaskMind - AI Assistant", page_icon="🤖")
st.title("🧠 TaskMind AI Assistant")
st.markdown("Ask me anything! I understand tasks, general queries, and real-time questions.")

# Session state
if "chat_logs" not in st.session_state:
    st.session_state.chat_logs = []

user_input = st.text_input("📝 Enter your query:")

# Process query
if st.button("Submit") and user_input.strip():
    st.markdown("⏳ Processing your request...")
    result = results(Username, user_input)

    # Save chat log
    st.session_state.chat_logs.append({
        "user": user_input,
        "assistant": result
    })

    # Display result
    st.markdown("### 💬 Response:")
    if result["type"] == "text":
        st.write(result["content"])

    elif result["type"] == "image":
        if os.path.exists(result["path"]):
            st.image(result["path"], caption="Generated Image", use_container_width=True)
            with open(result["path"], "rb") as file:
                st.download_button(
                    label="⬇️ Download Image",
                    data=file,
                    file_name=os.path.basename(result["path"]),
                    mime="image/jpeg"
                )
        else:
            st.error("Image not found.")

    elif result["type"] == "map":
        st.markdown(f"[🗺️ Click here to open the map]({result['url']})", unsafe_allow_html=True)

# Chat history
if st.session_state.chat_logs:
    st.markdown("### 🗨️ Chat History:")
    for entry in st.session_state.chat_logs:
        st.write(f"**User:** {entry['user']}")
        assistant_reply = entry["assistant"]

        if assistant_reply["type"] == "text":
            st.write(f"**Assistant:** {assistant_reply['content']}")
        elif assistant_reply["type"] == "image":
            if os.path.exists(assistant_reply["path"]):
                st.image(assistant_reply["path"], caption="Generated Image", use_container_width=True)
                with open(assistant_reply["path"], "rb") as file:
                    st.download_button(
                        label=f"⬇️ Download Image ({os.path.basename(assistant_reply['path'])})",
                        data=file,
                        file_name=os.path.basename(assistant_reply["path"]),
                        mime="image/jpeg"
                    )
            else:
                st.write("**Assistant:** [Image not found]")
        elif assistant_reply["type"] == "map":
            st.markdown(f"**Assistant:** [🗺️ Open Map]({assistant_reply['url']})", unsafe_allow_html=True)
