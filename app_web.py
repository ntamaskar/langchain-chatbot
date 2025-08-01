
import os
import streamlit as st
from langchain_openai import OpenAI

# 🔑 Get API key from environment
import streamlit as st
api_key = st.secrets["OPENAI_API_KEY"]

if not api_key:
    st.error("⚠️ OPENAI_API_KEY not set. Set it before running.")
    st.stop()

# 🤖 Initialize OpenAI LLM
llm = OpenAI(openai_api_key=api_key, temperature=0.7, model="gpt-3.5-turbo-instruct")

# 🎨 Custom CSS for better styling
st.markdown("""
    <style>
        body { background-color: #f9f9f9; }
        .stChatMessage.user { background: #e0f7fa; border-radius: 10px; padding: 10px; }
        .stChatMessage.assistant { background: #fff3e0; border-radius: 10px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# 🏷️ App Title
st.title("💬 Natty's Smart Assistant")
st.caption("Enhanced UI - LangChain + Streamlit + OpenAI")

# 🧠 Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 🖼️ Display chat history
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ⌨️ User input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message immediately
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # ✅ Generate assistant response (with proper error handling)
    try:
        raw_response = llm.invoke(user_input)  # returns an object
        response = raw_response if isinstance(raw_response, str) else str(raw_response)
    except Exception as e:
        response = f"⚠️ Sorry, there was an error contacting OpenAI: {str(e)}"

    # Save and display assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)
