from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
PERSIST_DIRECTORY = "./chroma_db"  # Note: Relative to where you RUN the app
LLM_MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"

# --- 1. Initialize Models and Vector Store (Global) ---
# We load these once when the API starts
try:
    llm = Ollama(model=LLM_MODEL)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 5}
    )
    print("--- Models and Vector Store Loaded ---")

except Exception as e:
    print(f"--- Error loading models or vector store: {e} ---")
    llm, retriever = None, None

# --- 2. Define the RAG Prompt and lightweight chain ---
# This prompt is slightly different, designed for simple string context injection
template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Be concise.

CONTEXT:
{context}

QUESTION:
{input}

ANSWER:
"""
prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()


def _format_docs(docs) -> str:
    """Join page contents of retrieved docs into a single context string."""
    try:
        return "\n\n".join(getattr(d, "page_content", "") for d in docs if getattr(d, "page_content", ""))
    except Exception:
        return ""

print("--- RAG Components Ready ---")


# --- 3. Define FastAPI App and Data Models ---
app = FastAPI()


class QueryRequest(BaseModel):
    """Pydantic model for the input query."""
    query: str


class SourceDocument(BaseModel):
    """Pydantic model for a single source document."""
    url: str
    content_snippet: str


class QueryResponse(BaseModel):
    """Pydantic model for the API response."""
    answer: str
    sources: List[SourceDocument]


# --- 4. Define the API Endpoint ---
@app.post("/query")
async def handle_query(request: QueryRequest) -> QueryResponse:
    """
    The main endpoint to ask questions to the RAG pipeline.
    Accepts: {"query": "Your question here"}
    Returns: {"answer": "...", "sources": [...]}
    """
    if llm is None or retriever is None:
        return QueryResponse(
            answer="Error: RAG pipeline not initialized. Check server logs.",
            sources=[]
        )

    # 1. Retrieve documents
    try:
        docs = retriever.invoke(request.query)
    except Exception:
        # Fallback to the vectorstore retriever's legacy API if needed
        try:
            docs = retriever.get_relevant_documents(request.query)
        except Exception:
            docs = []

    # 2. Build context and generate answer
    context_text = _format_docs(docs)
    try:
        answer = (prompt | llm | parser).invoke({
            "input": request.query,
            "context": context_text
        })
    except Exception as e:
        answer = f"Error generating answer: {e}"

    # 3. Format the sources
    response_sources: List[SourceDocument] = []
    for doc in docs or []:
        try:
            url = (doc.metadata or {}).get('source') or (doc.metadata or {}).get('url') or 'Unknown'
            snippet = (getattr(doc, 'page_content', '') or '')[:200] + '...'
            response_sources.append(SourceDocument(url=url, content_snippet=snippet))
        except Exception:
            continue

    return QueryResponse(
        answer=answer if isinstance(answer, str) else str(answer),
        sources=response_sources
    )


@app.get("/")
def read_root():
    return {"message": "RAG-Wiki-v2 API is running. POST to /query to ask questions."}