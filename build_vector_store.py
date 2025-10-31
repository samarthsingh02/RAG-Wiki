import jsonlines
import time
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
ARTICLES_FILE = "wiki_crawler/articles.jsonl"  # Path to your crawled data
PERSIST_DIRECTORY = "./chroma_db"             # Where to save the vector store
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Models ---
# Initialize the embedding model (using the one you pulled)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# --- Step 2: Data Loading & Chunking ---
print("--- Step 2: Loading and Chunking Documents ---")

documents = []
start_time = time.time()

# Open and read the .jsonl file
try:
    with jsonlines.open(ARTICLES_FILE) as reader:
        for article in reader:
            # Create a LangChain Document object for each article
            # We store the URL in the metadata
            doc = Document(
                page_content=article['text'],
                metadata={'source': article['url']}
            )
            documents.append(doc)
except FileNotFoundError:
    print(f"Error: The file {ARTICLES_FILE} was not found.")
    print("Please make sure you have run the Scrapy crawler (Step 1) first.")
    exit()

load_time = time.time() - start_time
print(f"Loaded {len(documents)} documents in {load_time:.2f} seconds.")

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Split all documents into chunks
all_chunks = text_splitter.split_documents(documents)

split_time = time.time() - start_time - load_time
print(f"Split {len(documents)} documents into {len(all_chunks)} chunks in {split_time:.2f} seconds.")

# --- Step 3: Embedding & Indexing (Vector Store) ---
print("\n--- Step 3: Embedding Chunks and Building Vector Store ---")
print(f"This will take some time... (Embedding {len(all_chunks)} chunks)")

start_embed_time = time.time()

# Create the persistent vector store
# This one command does all the work:
# 1. Calls the embedding_model for each chunk
# 2. Stores the resulting vector in ChromaDB
# 3. Persists the database to disk at PERSIST_DIRECTORY
db = Chroma.from_documents(
    documents=all_chunks,
    embedding=embedding_model,
    persist_directory=PERSIST_DIRECTORY
)

embed_time = time.time() - start_embed_time
print(f"Successfully built and persisted vector store in {embed_time:.2f} seconds.")
print(f"Vector store saved to: {PERSIST_DIRECTORY}")
print("\n--- Setup Complete! ---")