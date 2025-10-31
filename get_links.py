# Save this as get_links.py in your project root
import requests
from bs4 import BeautifulSoup
import os

# The Wikipedia category page for Deep Learning
CATEGORY_URL = "https://en.wikipedia.org/wiki/Category:Deep_learning"
OUTPUT_FILE = os.path.join('wiki_crawler', 'urls.txt')

# --- NEW ---
# Define a browser-like header to avoid the 403 Forbidden error
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
# -----------

print(f"--- Scraping category page: {CATEGORY_URL} ---")

try:
    # --- MODIFIED ---
    # Pass the headers with the request
    response = requests.get(CATEGORY_URL, headers=HEADERS)
    # ----------------

    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the div that contains the list of pages
    page_list_div = soup.find('div', id='mw-pages')

    if not page_list_div:
        print("Error: Could not find the 'mw-pages' div on the category page.")
        exit(1)

    # Find all <a> tags (links) within that div
    links = page_list_div.find_all('a')

    urls_to_save = []
    for link in links:
        href = link.get('href')
        # Ensure it's a valid article link and not a subcategory
        if href and href.startswith('/wiki/') and ':' not in href:
            full_url = f"https://en.wikipedia.org{href}"
            urls_to_save.append(full_url)

    # Save the links to the output file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for url in urls_to_save:
            f.write(f"{url}\n")

    print(f"--- Success! Saved {len(urls_to_save)} URLs to {OUTPUT_FILE} ---")

except requests.RequestException as e:
    print(f"Error making request: {e}")