import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ── Step 1: Load the PDF ──────────────────────────────────────
print("📄 Loading PDF...")
loader = PyPDFLoader("documents/sample.pdf")  # ← change to your filename
pages = loader.load()
print(f"✅ Loaded {len(pages)} pages")

# ── Step 2: Split into chunks ─────────────────────────────────
print("\n✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # each chunk = 500 characters
    chunk_overlap=50      # 50 character overlap between chunks
)
chunks = splitter.split_documents(pages)
print(f"✅ Created {len(chunks)} chunks")

# ── Step 3: Create Embeddings + Store in ChromaDB ─────────────
print("\n🧠 Creating embeddings and storing in ChromaDB...")
print("   (First run downloads the embedding model ~90MB, be patient...)")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"   # saved locally in this folder
)

print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
print("\n🎉 Ingestion complete! Your PDF is ready to be queried.")