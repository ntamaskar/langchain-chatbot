# app_web.py — chat + conversational memory + PDF upload (RAG) + live web search + version label

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

# Live web search (Tavily)
try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None  # will handle gracefully below


# ========= App / Version =========
APP_VERSION = os.getenv("APP_VERSION") or datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# ========= Keys =========
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("⚠️ OPENAI_API_KEY not found. Add it in Streamlit Secrets (cloud) or your .env (local).")
    st.stop()

TAVILY_KEY = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_KEY) if (TAVILY_KEY and TavilyClient) else None

# ========= Model =========
chat = ChatOpenAI(
    openai_api_key=API_KEY,
    model="gpt-3.5-turbo",
    temperature=0.7,
)

# ========= UI Header =========
st.title("💬 Natty's Smart Assistant")
st.caption(f"LangChain + Streamlit • memory + docs + live web • Version: {APP_VERSION}")

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
    ("system",
     "You are a helpful, concise assistant. "
     "Use the conversation history and any provided web context and/or uploaded docs context to answer. "
     "If you used web context or uploaded docs, mention it briefly."),
    MessagesPlaceholder("history"),
    ("human", "{input}\n\n[Optional web context]\n{web_context}")
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getbuffer())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs += loader.load()

    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    return FAISS.from_documents(chunks, embeddings)

if uploaded:
    needs_index = ("vs" not in st.session_state) or \
                  ("vs_file_count" not in st.session_state) or \
                  (st.session_state.vs_file_count != len(uploaded))
    if needs_index:
        with st.sidebar.status("Indexing PDFs…", expanded=False):
            st.session_state.vs = build_index_from_pdfs(uploaded)
            st.session_state.vs_file_count = len(uploaded)
        st.sidebar.success(f"Indexed {st.session_state.vs_file_count} file(s) ✅")

# ========= Sidebar: Live Web Search =========
st.sidebar.header("🌐 Live Web")
live_search = st.sidebar.checkbox("Use live web search for current events", value=False)
if live_search and not tavily:
    st.sidebar.warning("Add TAVILY_API_KEY in Secrets to enable live web search.")

def fetch_web_context(query: str, max_results: int = 5) -> str:
    if not tavily:
        return ""
    try:
        r = tavily.search(query=query, max_results=max_results)
        snippets = []
        for item in r.get("results", []):
            title = item.get("title", "")
            snip = item.get("content", "")
            url = item.get("url", "")
            if title or snip:
                snippets.append(f"- {title}\n  {snip}\n  Source: {url}")
        context = "\n".join(snippets)
        return context[:6000]
    except Exception as e:
        st.sidebar.error(f"Web search failed: {e}")
        return ""

# ========= Render chat so far =========
for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

# ========= Chat input =========
user_input = st.chat_input("Ask me anything…")

if user_input:
    st.session_state.history.append(HumanMessage(user_input))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    try:
        web_context = fetch_web_context(user_input, max_results=5) if live_search else ""

        if use_docs and "vs" in st.session_state:
            retriever = st.session_state.vs.as_retriever(search_type="similarity", k=4)
            qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever)
            base_answer = qa.invoke({"query": user_input})["result"]

            chain = prompt | chat
            result = chain.invoke({
                "history": st.session_state.history,
                "input": f"User asked: {user_input}\nRAG answer (from uploaded docs): {base_answer}",
                "web_context": web_context
            })
            answer = result.content
        else:
            chain = prompt | chat
            result = chain.invoke({
                "history": st.session_state.history,
                "input": user_input,
                "web_context": web_context
            })
            answer = result.content

    except Exception as e:
        answer = f"⚠️ Error contacting the model: {e}"

    st.session_state.history.append(AIMessage(answer))
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(answer)
