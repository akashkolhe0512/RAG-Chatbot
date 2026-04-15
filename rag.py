import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Step 1: Load the same embeddings + ChromaDB ───────────────
print("🔌 Connecting to ChromaDB...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)
print("✅ Connected!")

# ── Step 2: Create a Retriever ────────────────────────────────
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ── Step 3: Load Groq LLM ─────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ── Step 4: Create Prompt Template ───────────────────────────
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context from a document
to answer the question. If you don't know the answer from the context,
say "I don't find this in the document."

Context:
{context}

Question: {question}

Answer:
""")

# ── Step 5: Helper to format retrieved chunks ─────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Step 6: Build RAG Chain (modern LCEL style) ───────────────
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ── Step 7: Ask Questions! ────────────────────────────────────
print("\n🤖 RAG Chatbot Ready! Type 'exit' to quit\n")

while True:
    question = input("You: ")
    if question.lower() == "exit":
        break

    # Get answer
    answer = rag_chain.invoke(question)
    print(f"\n🤖 Answer: {answer}")

    # Show source chunks used
    source_docs = retriever.invoke(question)
    print("\n📄 Sources used:")
    for i, doc in enumerate(source_docs):
        print(f"  Chunk {i+1} (Page {doc.metadata.get('page', '?')}): {doc.page_content[:100]}...")

    print("\n" + "─"*50 + "\n")