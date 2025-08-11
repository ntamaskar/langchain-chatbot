# app_web.py — chat + conversational memory + PDF upload (RAG) + version label

import os
import tempfile
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ========== Config / Version ==========
from datetime import datetime

APP_VERSION = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
st.sidebar.markdown(f"**Version:** {APP_VERSION}")


# ========== Keys ==========
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("⚠️ OPENAI_API_KEY not found. Add it in Streamlit Secrets or your .env.")
    st.stop()

# ========== Model ==========
chat = ChatOpenAI(
    openai_api_key=API_KEY,
    model="gpt-3.5-turbo",
    temperature=0.7,
)

# ========== UI Header ==========
st.title("💬 Natty's Smart Assistant")
st.caption(f"LangChain + Streamlit • memory + docs • Version: {APP_VERSION}")

# Optional style
st.markdown("""
<style>
    body { background-color: #f9f9f9; }
    .stChatMessage.user { background: #e0f7fa; border-radius: 10px; padding: 10px; }
    .stChatMessage.assistant { background: #fff3e0; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ========== Conversational Memory ==========
if "history" not in st.session_state:
    st.session_state.history = []  # list[HumanMessage | AIMessage]

# Prompt that includes history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, concise assistant. If you use uploaded documents, cite what you used in plain English."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

# ========== Sidebar: Upload & Index PDFs ==========
st.sidebar.header("📄 Ask Your Docs")
uploaded_files = st.sidebar.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
use_docs = st.sidebar.checkbox("Use uploaded docs for answers", value=False)

# Build/refresh vector store when new files are uploaded
def build_index_from_pdfs(files):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for f in files:
        # Save to a temp path (Streamlit Cloud allows writing to /tmp)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getbuffer())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs += loader.load()
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(api_key=API_KEY)
    return FAISS.from_documents(chunks, embeddings)

# Cache the vectorstore for the current session
if "vs" not in st.session_state and uploaded_files:
    with st.sidebar.status("Indexing PDFs…", expanded=False):
        st.session_state.vs = build_index_from_pdfs(uploaded_files)
        st.sidebar.success("Indexed ✅")

# ========== Render chat so far ==========
for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

# ========== Chat input ==========
user_input = st.chat_input("Ask me anything…")

if user_input:
    # Show + store user turn
    st.session_state.history.append(HumanMessage(user_input))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    try:
        if use_docs and "vs" in st.session_state:
            retriever = st.session_state.vs.as_retriever(search_type="similarity", k=4)
            qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever)
            answer = qa.invoke({"query": user_input})["result"]
        else:
            # plain chat with memory
            chain = prompt | chat
            result = chain.invoke({"history": st.session_state.history, "input": user_input})
            answer = result.content
    except Exception as e:
        answer = f"⚠️ Error contacting the model: {e}"

    # Save + display assistant turn
    st.session_state.history.append(AIMessage(answer))
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(answer)
