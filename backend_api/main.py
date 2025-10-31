import sys
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# --- Imports we know WORK in your environment ---
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever  # This one is in -community
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# --- Configuration ---
PERSIST_DIRECTORY = "./chroma_db"
LLM_MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"

# --- 1. Load All Models and Retrievers (Global) ---
print("--- Loading Models and Vector Store ---")
try:
    # LLM and Embedding Model
    llm = OllamaLLM(model=LLM_MODEL)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # ChromaDB Vector Store
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    chroma_retriever = db.as_retriever(search_kwargs={'k': 25}) # Get Top 25 vector results
    print("--- ChromaDB Loaded ---")

    # Load all docs from Chroma for BM25
    print("--- Loading docs from Chroma for BM25 ---")
    # We need both the content and the metadata to pass to BM25
    all_docs_data = db.get(include=["documents", "metadatas"])
    all_texts = all_docs_data['documents']
    all_metadatas = all_docs_data['metadatas']

    # BM25 Keyword Retriever
    bm25_retriever = BM25Retriever.from_texts(
        all_texts,
        metadatas=all_metadatas
    )
    bm25_retriever.k = 25  # Get Top 25 keyword results
    print("--- BM25 Retriever Initialized ---")

    # Cross-Encoder Re-ranker Model
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("--- Cross-Encoder Re-ranker Loaded ---")

except Exception as e:
    print(f"--- FATAL ERROR loading models: {e} ---")
    sys.exit(1)


# --- 2. Define the RAG Prompt and Chain (same as before) ---
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

def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content)

print("--- RAG Components Ready ---")


# --- 3. Define FastAPI App and Data Models (same as before) ---
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class SourceDocument(BaseModel):
    url: str
    content_snippet: str
    score: float  # We'll add the re-ranker's score now

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]


# --- 4. Define the NEW API Endpoint Logic (MANUAL HYBRID SEARCH) ---
@app.post("/query")
async def handle_query(request: QueryRequest) -> QueryResponse:
    """
    The main endpoint.
    Manually performs 2-stage Hybrid Search + Re-ranking.
    """

    # === STAGE 1: HYBRID SEARCH (MANUAL) ===
    # 1. Get vector search results
    try:
        vector_docs = chroma_retriever.invoke(request.query)
    except Exception:
        vector_docs = chroma_retriever.get_relevant_documents(request.query) # legacy

    # 2. Get keyword search results
    try:
        keyword_docs = bm25_retriever.invoke(request.query)
    except Exception:
        keyword_docs = bm25_retriever.get_relevant_documents(request.query) # legacy

    # 3. Combine and de-duplicate the results
    combined_docs = {doc.page_content: doc for doc in vector_docs + keyword_docs}.values()
    retrieved_docs = list(combined_docs)

    if not retrieved_docs:
        return QueryResponse(answer="No relevant documents found.", sources=[])

    # === STAGE 2: RE-RANKING (Find the Best 5) ===
    # Create pairs of [query, document_text] for the Cross-Encoder
    query_doc_pairs = [[request.query, doc.page_content] for doc in retrieved_docs]

    # Run the Cross-Encoder to get relevance scores
    scores = cross_encoder.predict(query_doc_pairs)

    # Combine docs with their new scores
    doc_scores = list(zip(retrieved_docs, scores))

    # Sort by the new score in descending order
    doc_scores_sorted = sorted(doc_scores, key=lambda x: x[1], reverse=True)

    # Get the Top 5 re-ranked documents
    top_5_docs_with_scores = doc_scores_sorted[:5]
    top_5_docs = [doc for doc, score in top_5_docs_with_scores]

    # === STAGE 3: GENERATE ANSWER (Same as before) ===
    context_text = _format_docs(top_5_docs)

    try:
        answer = (prompt | llm | parser).invoke({
            "input": request.query,
            "context": context_text
        })
    except Exception as e:
        answer = f"Error generating answer: {e}"

    # Format the sources, now with the score
    response_sources: List[SourceDocument] = []
    for doc, score in top_5_docs_with_scores:
        response_sources.append(SourceDocument(
            url=doc.metadata.get('source', 'Unknown'),
            content_snippet=doc.page_content[:200] + "...",
            score=float(score)
        ))

    return QueryResponse(
        answer=answer if isinstance(answer, str) else str(answer),
        sources=response_sources
    )

@app.get("/")
def read_root():
    return {"message": "RAG-Wiki-v2 API (Manual Hybrid Search + Re-ranker) is running."}