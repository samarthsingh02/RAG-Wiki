import duckdb
import os

DB_FILE = "processed_pages.db"

if not os.path.exists(DB_FILE):
    print(f"Error: Database file not found at {DB_FILE}")
    print("Please run the Dagster orchestrator (orchestrator.py) at least once to create it.")
else:
    # Connect to the database (read-only)
    conn = duckdb.connect(DB_FILE, read_only=True)

    try:
        # Query the table for all URLs
        urls = conn.execute("SELECT url FROM processed_pages").fetchall()

        print(f"--- Found {len(urls)} Articles in the Processing Database ---")

        # Print each URL
        for i, url_tuple in enumerate(urls):
            print(f"{i + 1}: {url_tuple[0]}")

    except duckdb.CatalogException:
        print("Error: The 'processed_pages' table does not exist.")
        print("This might mean the orchestrator ran but failed before creating the table.")
    finally:
        conn.close()