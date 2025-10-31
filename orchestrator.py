import sys
import jsonlines
import subprocess
import duckdb
from dagster import asset, DagsterInstance, Definitions, materialize

from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
ARTICLES_FILE = "wiki_crawler/articles.jsonl"
CRAWLER_OUTPUT_FILENAME = "articles.jsonl"
PROCESSED_DB_FILE = "processed_pages.db"
PERSIST_DIRECTORY = "./chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Initialize models (Dagster assets load these in their own processes)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")


def get_db_conn():
    """Helper function to connect to DuckDB and create table if it doesn't exist."""
    conn = duckdb.connect(PROCESSED_DB_FILE)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_pages (
            url TEXT PRIMARY KEY,
            last_modified TEXT
        )
    """
    )
    return conn


@asset
def crawled_articles_file() -> str:
    """
    Dagster Asset: Runs the Scrapy crawler to fetch articles and timestamps.
    Outputs the path to the resulting articles.jsonl file.
    """
    print("--- Running Scrapy Crawler ---")
    # We must run Scrapy as a subprocess from its own directory
    # The -O flag overwrites the file, which is what we want
    subprocess.run(
        [sys.executable, "-m", "scrapy", "crawl", "wiki_spider", "-O", CRAWLER_OUTPUT_FILENAME],
        cwd="./wiki_crawler",  # Run from the 'wiki_crawler' directory
        check=True
    )
    print(f"--- Crawler finished. Data saved to {ARTICLES_FILE} ---")
    return ARTICLES_FILE


@asset(deps=[crawled_articles_file])
def intelligent_vector_store():
    """
    Dagster Asset: Reads the crawled articles, intelligently checks against
    a local DB (DuckDB) to find new/updated pages, and updates the
    Chroma vector store.
    """
    print("--- Starting Intelligent Vector Store Update ---")

    # 1. Connect to our tracking DB and ChromaDB
    tracker_conn = get_db_conn()
    chroma_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )

    # 2. Find which articles are new or updated
    print("--- Checking for new or updated articles ---")
    processed_pages = tracker_conn.execute("SELECT * FROM processed_pages").fetchdf()
    processed_dict = processed_pages.set_index('url')['last_modified'].to_dict()

    articles_to_process = []
    with jsonlines.open(ARTICLES_FILE) as reader:
        for article in reader:
            url = article['url']
            new_last_mod = article['last_modified']

            if url not in processed_dict or processed_dict[url] != new_last_mod:
                articles_to_process.append(article)

    if not articles_to_process:
        print("--- No new or updated articles found. Vector store is up-to-date. ---")
        return "Vector store is up-to-date."

    print(f"--- Found {len(articles_to_process)} new/updated articles to process. ---")

    # 3. Process the new/updated articles
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    all_new_chunks = []
    all_new_chunk_ids = []
    urls_to_delete = []

    for article in articles_to_process:
        url = article['url']
        # Add to list of URLs to delete. We'll wipe all old chunks for this
        # page to ensure no old data remains, then add the new chunks.
        urls_to_delete.append(url)

        # Create LangChain Document
        doc = Document(
            page_content=article['text'],
            metadata={'source': article['url']}
        )

        # Split into chunks
        chunks = text_splitter.split_documents([doc])

        # Create unique, repeatable IDs for each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"{url}#chunk{i}"
            all_new_chunks.append(chunk)
            all_new_chunk_ids.append(chunk_id)

    # 4. Update ChromaDB (Delete old, add new)
    if urls_to_delete:
        print(f"--- Deleting old chunks for {len(urls_to_delete)} updated articles ---")
        # We must get all chunk IDs for the URLs we are updating
        existing_chunks = chroma_db.get(where={"source": {"$in": urls_to_delete}})
        if existing_chunks and existing_chunks['ids']:
            chroma_db.delete(ids=existing_chunks['ids'])

    if all_new_chunks:
        print(f"--- Adding {len(all_new_chunks)} new chunks to vector store in batches ---")

        # Define a safe batch size (well under the 5461 limit)
        BATCH_SIZE = 4000

        for i in range(0, len(all_new_chunks), BATCH_SIZE):
            # Find the end of this batch
            end_index = min(i + BATCH_SIZE, len(all_new_chunks))

            # Get the mini-batch of documents and their IDs
            batch_docs = all_new_chunks[i:end_index]
            batch_ids = all_new_chunk_ids[i:end_index]

            # Log to Dagster
            print(f"--- Processing batch {i // BATCH_SIZE + 1}: Adding {len(batch_docs)} chunks ---")

            # Add this small batch to ChromaDB
            chroma_db.add_documents(documents=batch_docs, ids=batch_ids)

        print("--- ChromaDB update complete. ---")

    # 5. Update our tracking DB
    print("--- Updating processing tracker DB ---")
    update_data = [(article['url'], article['last_modified']) for article in articles_to_process]

    # Use DuckDB's fast executemany with a temporary table
    tracker_conn.executemany("INSERT OR REPLACE INTO processed_pages (url, last_modified) VALUES (?, ?)", update_data)
    tracker_conn.close()

    print(f"--- Successfully processed {len(articles_to_process)} articles. ---")
    return f"Successfully processed {len(articles_to_process)} articles."


# This Definitions object tells Dagster what to load
defs = Definitions(
    assets=[crawled_articles_file, intelligent_vector_store],
)


# This allows you to run the script directly for testing
if __name__ == "__main__":
    print("Running Dagster pipeline one-off...")
    # Note: This uses a simple in-memory instance.
    # For production, you'd use 'dagster dev'
    result = materialize(
        assets=[crawled_articles_file, intelligent_vector_store],
        instance=DagsterInstance.ephemeral(),
    )
    print(result.get_output_for_node("intelligent_vector_store"))