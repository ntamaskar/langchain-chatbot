# app_web.py — upgraded to conversational memory

import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 🔑 Load API key (Cloud first, local fallback)
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("⚠️ OPENAI_API_KEY not found. Set it in Streamlit Secrets or your .env.")
    st.stop()

# 🤖 Chat model (use chat model for memory-aware convos)
chat = ChatOpenAI(
    openai_api_key=API_KEY,
    model="gpt-3.5-turbo",   # or "gpt-4o-mini" if you switch later
    temperature=0.7,
)

# 🎨 Custom CSS (kept from your version)
st.markdown("""
    <style>
        body { background-color: #f9f9f9; }
        .stChatMessage.user { background: #e0f7fa; border-radius: 10px; padding: 10px; }
        .stChatMessage.assistant { background: #fff3e0; border-radius: 10px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# 🏷️ App Title
st.title("💬 Natty's Smart Assistant")
st.caption("Enhanced UI - LangChain + Streamlit + OpenAI (with memory)")

# 🧠 LangChain-style memory
if "history" not in st.session_state:
    st.session_state.history = []  # list of BaseMessage: HumanMessage / AIMessage

# 🧩 Prompt with history placeholder so past turns are included
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, concise assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

# 🖼️ Display chat history with avatars
for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

# ⌨️ User input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # show + store user turn
    st.session_state.history.append(HumanMessage(user_input))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # generate with full history context
    try:
        chain = prompt | chat
        result = chain.invoke({
            "history": st.session_state.history,
            "input": user_input
        })
        answer = result.content
    except Exception as e:
        answer = f"⚠️ Sorry, there was an error contacting OpenAI: {e}"

    # store + show assistant turn
    st.session_state.history.append(AIMessage(answer))
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(answer)
