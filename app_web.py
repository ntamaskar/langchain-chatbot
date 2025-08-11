# app_web.py — chat + conversational memory + PDF upload (RAG) + version label

import os
import tempfile
from datetime import datetime
import streamlit as st

# LangChain core
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# OpenAI (via langchain-openai)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAG bits (community)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


# ========= App / Version =========
# Option A (auto): show current UTC timestamp on each run
APP_VERSION = os.getenv("APP_VERSION") or datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# ========= Keys =========
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("⚠️ OPENAI_API_KEY not found. Add it in Streamlit Secrets (cloud) or your .env (local).")
    st.stop()

# ========= Model =========
chat = ChatOpenAI(
    openai_api_key=API_KEY,
    model="gpt-3.5-turbo",  # fine balance of quality/latency/cost
    temperature=0.7,
)

# ========= UI Header =========
st.title("💬 Natty's Smart Assistant")
st.caption(f"LangChain + Streamlit • memory + docs • Version: {APP_VERSION}")

# (optional) simple styling to keep your look
st.markdown("""
<style>
  body { background-color: #f9f9f9; }
  .stChatMessage.user { background: #e0f7fa; border-radius: 10px; padding: 10px; }
  .stChatMessage.assistant { background: #fff3e0; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ========= Conversational Memory =========
if "history" not in st.session_state:
    st.session_state.history = []  # list[HumanMessage | AIMessage]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, concise assistant. "
               "If you use information from uploaded documents, say so in plain English."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

# ========= Sidebar: Upload + Index PDFs =========
st.sidebar.header("📄 Ask Your Docs")
uploaded = st.sidebar.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
use_docs = st.sidebar.checkbox("Use uploaded docs for answers", value=False)

def build_index_from_pdfs(files):
    """Load PDFs, chunk, embed, and return a FAISS vector store."""
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for f in files:
        # Streamlit allows writing to /tmp in the cloud
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getbuffer())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs += loader.load()

    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    return FAISS.from_documents(chunks, embeddings)

# Build/refresh the vector index when new files are uploaded
if uploaded:
    # Rebuild only if we don't have one yet OR file list changed in size
    needs_index = ("vs" not in st.session_state) or \
                  ("vs_file_count" not in st.session_state) or \
                  (st.session_state.vs_file_count != len(uploaded))

    if needs_index:
        with st.sidebar.status("Indexing PDFs…", expanded=False):
            st.session_state.vs = build_index_from_pdfs(uploaded)
            st.session_state.vs_file_count = len(uploaded)
            st.sidebar.success(f"Indexed {st.session_state.vs_file_count} file(s) ✅")

# ========= Render chat so far =========
for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

# ========= Chat input =========
user_input = st.chat_input("Ask me anything…")

if user_input:
    # Show + store user turn
    st.session_state.history.append(HumanMessage(user_input))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    try:
        if use_docs and "vs" in st.session_state:
            # RAG path: retrieve from the uploaded docs
            retriever = st.session_state.vs.as_retriever(search_type="similarity", k=4)
            qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever)
            answer = qa.invoke({"query": user_input})["result"]
        else:
            # Plain chat with conversational memory
            chain = prompt | chat
            result = chain.invoke({"history": st.session_state.history, "input": user_input})
            answer = result.content
    except Exception as e:
        answer = f"⚠️ Error contacting the model: {e}"

    # Save + display assistant turn
    st.session_state.history.append(AIMessage(answer))
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(answer)
