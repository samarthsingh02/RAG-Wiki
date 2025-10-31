import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup


class WikiSpider(CrawlSpider):
    """
    A Scrapy spider to crawl Wikipedia pages.

    This spider starts at a seed URL, follows links within Wikipedia,
    and parses the article text from pages it visits.
    """

    # 1. Spider configuration
    name = 'wiki_spider'  # The unique name to call this spider
    allowed_domains = ['en.wikipedia.org']  # Restricts crawling to this domain
    start_urls = ['https://en.wikipedia.org/wiki/Deep_learning']  # The starting point

    # 2. Crawling Rules
    # This is the magic of Scrapy. It defines how to follow links.
    rules = (
        # Rule 1: Follow relevant links and parse them with 'parse_article'
        # - LinkExtractor finds links. We restrict it to:
        #   - Only links starting with '/wiki/'
        #   - Not links containing ':' (which are special pages like 'File:', 'Talk:')
        # - 'callback="parse_article"': When a link is followed, call this method.
        # - 'follow=True': Continue to follow links from the pages it finds.
        Rule(
            LinkExtractor(allow=r'/wiki/', deny=r'/wiki/.*:.*'),
            callback='parse_article',
            follow=True
        ),
    )

    # 3. Custom Settings (Good Practice)
    # We limit the crawl to 500 pages (as per the plan) and set a depth limit.
    custom_settings = {
        'CLOSESPIDER_PAGECOUNT': 500,  # Stop after 500 pages are crawled
        'DEPTH_LIMIT': 5,              # Don't go deeper than 5 links from the start page
        'USER_AGENT': 'RAGWikiCrawler/1.0 (samarthsingh02.official@gmail.com)' # Be a good web citizen
    }

    def parse_article(self, response):
        """
        This method is called for every page Scrapy crawls.
        It extracts the text content and yields it as a Scrapy item.
        """

        # 1. Extract the clean paragraph text (same as before)
        paragraphs = response.css('div#mw-content-text p::text').getall()
        all_paragraph_text = ' '.join(p.strip() for p in paragraphs if p.strip())

        # 2. Extract the "last modified" timestamp
        # We use BeautifulSoup to parse the footer for the specific list item
        soup = BeautifulSoup(response.body, 'html.parser')
        last_mod_element = soup.select_one('li#footer-info-lastmod')
        last_modified_text = ""
        if last_mod_element:
            # The text is like "This page was last modified on 31 October 2025, at 07:49."
            # We just store the whole string for simplicity.
            last_modified_text = last_mod_element.get_text(strip=True)

        # 3. Yield the enhanced data (a Scrapy "Item")
        if all_paragraph_text:
            yield {
                'url': response.url,
                'text': all_paragraph_text,
                'last_modified': last_modified_text  # <-- NEWLY ADDED
            }