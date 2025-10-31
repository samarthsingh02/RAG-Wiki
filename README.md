# RAG-Wiki: An End-to-End Wikipedia RAG Assistant

RAG-Wiki is a complete, end-to-end Retrieval-Augmented Generation (RAG) system built in Python. It allows a user to ask questions in natural language and receive answers sourced *exclusively* from a curated set of Wikipedia articles.

This project is a high-performance demo built in two phases:

1.  **Phase 1 (The Prototype):** A simple, local-first RAG application using Streamlit and a basic vector store. (See `build_vector_store.py`)
2.  **Phase 2 (The Upgrade):** A production-grade, decoupled architecture featuring an automated data pipeline, an advanced hybrid-search API, and a quantitative evaluation framework.

## 🏛️ Architecture

This system is decoupled into two main pipelines:

#### 1\. Automated Data Ingestion Pipeline (Managed by Dagster)

`Dagster (Orchestrator)` ➔ `Scrapy (Crawler)` ➔ `DuckDB (Tracker DB)` ➔ `Ollama (Embeddings)` ➔ `ChromaDB (Vector Store)`

  * The **Dagster** orchestrator runs the pipeline.
  * **Scrapy** crawls 500+ Wikipedia articles and extracts their "last modified" timestamp.
  * A **DuckDB** database is checked to see if an article is new or has been updated.
  * Only new/updated articles are processed, chunked, and embedded using **Ollama**.
  * The new vectors are added to the **ChromaDB** vector store.

#### 2\. RAG Query Pipeline (API-First)

`Streamlit (Frontend)` ➔ `FastAPI (Backend)` ➔ `[Hybrid Search + Re-ranking]` ➔ `Ollama (LLM)` ➔ `Streamlit (Frontend)`

  * The **Streamlit** UI sends a user's question to the backend.
  * The **FastAPI** backend receives the request.
  * **Hybrid Search** is performed:
    1.  `BM25Retriever` (keyword search) fetches the top 25 results.
    2.  `ChromaDB` (vector search) fetches the top 25 results.
  * **Re-ranking** is performed:
    1.  The \~50 combined results are fed to a `CrossEncoder` model.
    2.  This model re-ranks all results for relevance, and the true top 5 are selected.
  * The top 5 documents are passed to **Ollama (Mistral)** with a prompt.
  * The final answer is streamed back to the UI.

## ✨ Key Features

  * **Decoupled Architecture:** A robust FastAPI backend handles all AI logic, while a separate Streamlit frontend acts as the client.
  * **Automated Data Pipeline:** Uses **Dagster** to orchestrate an intelligent ingestion pipeline that only processes new or updated Wikipedia articles, saving hours of reprocessing.
  * **Advanced Hybrid Search:** Combines keyword (BM25) and semantic (Chroma) search to ensure both relevance and accuracy.
  * **High-Accuracy Re-ranking:** Implements a second-stage Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to re-rank hybrid search results, significantly improving the quality of the context provided to the LLM.
  * **Quantitative Evaluation:** Includes an evaluation script (`evaluate.py`) that uses the **RAGAs** framework to score the pipeline's performance on metrics like `faithfulness` and `answer_relevancy`.
  * **100% Local & Private:** The entire stack (including LLMs and embedding models) runs locally via **Ollama**, ensuring zero data leakage and zero API costs.

## 🛠️ Tech Stack

  * **Orchestration:** Dagster
  * **AI / LLMs:** Ollama, LangChain (`langchain-ollama`, `langchain-chroma`), Sentence-Transformers
  * **Backend:** FastAPI, Uvicorn
  * **Frontend:** Streamlit
  * **Data Storage:** ChromaDB (Vector Store), DuckDB (Metadata Tracker)
  * **Crawling:** Scrapy, BeautifulSoup
  * **Evaluation:** RAGAs, Pandas

## 🚀 Local Installation & Usage

Follow these steps to run the full application on your local machine.

### Prerequisites

  * Python 3.11
  * [Ollama](https://ollama.com/) installed and running.
  * **Docker Desktop** (for the final containerized step).

### Step 1: Initial Setup

1.  Clone this repository:
    ```bash
    git clone https://github.com/your-username/RAG-Wiki.git
    cd RAG-Wiki
    ```
2.  Create and activate a Python virtual environment:
    ```bash
    python -m venv .venv
    .\.venv\Scripts\Activate
    ```
3.  Install all required packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
4.  Pull the necessary models for Ollama:
    ```bash
    ollama pull mistral
    ollama pull nomic-embed-text
    ```

### Step 2: Run the Data Pipeline (First Time Only)

This step runs the Dagster pipeline to crawl Wikipedia and build your vector database.

1.  Use the `Run Dagster UI` configuration in PyCharm (or run `dagster dev -f orchestrator.py` in your terminal).
2.  Open **`http://127.0.0.1:3000`** in your browser.
3.  Click **"Overview"** and then **"Materialize all"**.
4.  Click **"Launch run"**.

> **Note:** This first run will take a long time (potentially hours) as it crawls 500+ pages and embeds thousands of text chunks. All future runs will be incremental and finish in seconds.

### Step 3: Run the Backend API

This starts the FastAPI server that handles all RAG logic.

1.  Use the `Run API Server` configuration in PyCharm.
2.  Wait for the log output to confirm:
    `INFO: Uvicorn running on http://127.0.0.1:8000`

### Step 4: Run the Frontend UI

This starts the Streamlit app you will interact with.

1.  In a **new terminal** (with your `.venv` active), run:
    ```bash
    streamlit run app.py
    ```
2.  Open **`http://127.0.0.1:8501`** in your browser to start chatting with your data.

## 🔬 Running the Evaluation

To prove that the RAG pipeline is effective, you can run the quantitative evaluation.

1.  Ensure your Backend API (Step 3) is **running**.

2.  In a new terminal (with your `.venv` active), run the `evaluate.py` script:

    ```bash
    python evaluate.py
    ```

3.  The script will take several minutes as it uses an LLM to "grade" its own answers. It will output a final report:

    ```
    --- Average Scores ---
    Faithfulness:     0.72
    Answer Relevancy: 0.52
    ```

## 🗺️ Roadmap / Future Work

  * **Containerization:** The project is set up for **Step 11: Containerize Everything**. By using the provided `Dockerfile`s and `docker-compose.yml`, the entire stack (API, UI, Ollama) can be launched with a single command: `docker-compose up --build`.
  * **Improve Scores:** The evaluation scores show room for improvement. This can be done by:
      * Tuning the prompt in `backend_api/main.py`.
      * Adjusting the weights of the `EnsembleRetriever`.
      * Implementing more advanced chunking strategies in `orchestrator.py`.
