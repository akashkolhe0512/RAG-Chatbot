import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Initialize FastAPI ────────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot API",
    description="Ask questions about your documents",
    version="1.0.0"
)

# ── Load RAG Pipeline once when server starts ─────────────────
print("🔌 Loading RAG pipeline...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context from a document
to answer the question. If you don't know the answer from the context,
say "I don't find this in the document."

Context:
{context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG pipeline ready!")

# ── Define Request & Response models ─────────────────────────
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]

# ── API Endpoints ─────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "RAG Chatbot API is running! Go to /docs to test it."}

@app.post("/chat", response_model=AnswerResponse)
def chat(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Get answer from RAG chain
    answer = rag_chain.invoke(request.question)

    # Get source chunks
    source_docs = retriever.invoke(request.question)
    sources = [
        f"Page {doc.metadata.get('page', '?')}: {doc.page_content[:100]}..."
        for doc in source_docs
    ]

    return AnswerResponse(
        question=request.question,
        answer=answer,
        sources=sources
    )