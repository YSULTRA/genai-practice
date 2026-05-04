import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# --- your existing modules ---
from text_splitter import splitter
from embedding import embedding_model # kept for initialization
from vector_store import vectorStore
from langchain_classic.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# PDF loader (standard LangChain)
from langchain_community.document_loaders import PyPDFLoader

# optional: preload your original docs
try:
    from data_loader import docs as initial_docs
except:
    initial_docs = []

# --- page config ---
st.set_page_config(page_title="PDF Chat AI", page_icon="📄", layout="wide")

# --- cool design CSS ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #e2e8f0;
}
.main-header {
    font-size: 2.5rem; font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.glass {
    background: rgba(30,41,59,0.6); backdrop-filter: blur(12px);
    border: 1px solid rgba(148,163,184,0.2); border-radius: 16px;
    padding: 1.5rem;
}
.stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📄 Chat with your PDFs</div>', unsafe_allow_html=True)
st.caption("Powered by your existing RAG pipeline — Gemini 3.1 Flash + MMR retriever")

# --- init session ---
if "initialized" not in st.session_state:
    # add your original docs once
    if initial_docs:
        splits = splitter.split_documents(initial_docs)
        vectorStore.add_documents(splits)

    st.session_state.retriever = vectorStore.as_retriever(
        search_type="mmr", search_kwargs={"k": 6, "fetch_k": 10, "lambda_mult": 0.25}
    )
    st.session_state.llm = init_chat_model(model="google_genai:gemini-3.1-flash-lite-preview")
    st.session_state.prompt = ChatPromptTemplate([
        ("system", "You are helpful ai assistnat which answer query answer using context given if you dont have answer just say i cannot answer this question"),
        ("human", "you context is {context} and your query is {query}")
    ])
    st.session_state.messages = []
    st.session_state.processed_files = set()
    st.session_state.initialized = True

# --- sidebar upload ---
with st.sidebar:
    st.markdown("### Upload PDFs")
    uploads = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True)

    if uploads:
        for f in uploads:
            if f.name not in st.session_state.processed_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    path = tmp.name

                with st.spinner(f"Indexing {f.name}..."):
                    loader = PyPDFLoader(path)
                    pdf_docs = loader.load()
                    splits = splitter.split_documents(pdf_docs)
                    vectorStore.add_documents(splits)
                    st.session_state.processed_files.add(f.name)
                os.unlink(path)
                st.success(f"Added {f.name}")

    st.markdown("---")
    st.metric("Files indexed", len(st.session_state.processed_files))
    if st.button("Clear chat"):
        st.session_state.messages = []

# --- chat UI ---
chat_container = st.container()
with chat_container:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    st.markdown('</div>', unsafe_allow_html=True)

query = st.chat_input("Ask anything about your PDFs...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = st.session_state.retriever.invoke(query)
            context = "\n\n".join([d.page_content for d in res])

            final_prompt = st.session_state.prompt.invoke({
                "context": context,
                "query": query
            })
            output = st.session_state.llm.invoke(final_prompt)

            # match your main.py output format
            try:
                answer = output.content[0]["text"]
            except:
                answer = output.content if isinstance(output.content, str) else str(output.content)

            st.write(answer)

            # show sources
            with st.expander("Sources"):
                for i, doc in enumerate(res[:3], 1):
                    st.caption(f"{i}. {doc.page_content[:200]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})