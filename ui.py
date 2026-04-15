import streamlit as st
import requests
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile

load_dotenv()

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────────
st.title("🤖 RAG Chatbot")
st.caption("Upload a PDF and ask questions about it!")

# ── Session State (stores chat history) ──────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

# ── Sidebar: PDF Upload ───────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and not st.session_state.pdf_loaded:
        with st.spinner("📄 Processing PDF..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load and chunk
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.split_documents(pages)

            # Embed and store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

            # Clean up temp file
            os.unlink(tmp_path)

            st.session_state.pdf_loaded = True
            st.session_state.vectorstore = vectorstore

        st.success(f"✅ Loaded {len(pages)} pages, {len(chunks)} chunks!")

    if st.session_state.pdf_loaded:
        st.info("✅ PDF ready! Ask questions below.")

    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Main Chat Area ────────────────────────────────────────────
if not st.session_state.pdf_loaded:
    st.info("👈 Please upload a PDF from the sidebar to get started!")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message:
                with st.expander("📄 View Sources"):
                    for src in message["sources"]:
                        st.caption(src)

    # Chat input
    if question := st.chat_input("Ask a question about your document..."):

        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Get answer from API
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/chat",
                        json={"question": question}
                    )
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]

                    st.write(answer)
                    with st.expander("📄 View Sources"):
                        for src in sources:
                            st.caption(src)

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}. Make sure app.py is running!")