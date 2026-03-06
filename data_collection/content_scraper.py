# Content scraper that extracts content from the URLS provided by GDELT
# Uses caching to prevent repeated scraping of repeated articles
# Uses multi-threading for speed and efficiency due to wait times from requests
# 

import os
import json
import hashlib
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict
from data_collection.models import NewsArticle


try:
    import trafilatura          # in case of failure
except ImportError:
    trafilatura = None
    print("ERROR: Unable to detect install of trafilatura. Exiting...")


class ContentScraper:
    """Scrapes full article text from URLs with caching and concurrency."""

    def __init__(self, cache_dir: str = "data/cache/articles",
                 max_threads: int = 10,
                 timeout: int = 15,
                 delay_between_requests: float = 0.2):
        """
        Args:
            cache_dir: Directory to store cached_url article content
            max_threads: Number of concurrent download threads
            timeout: HTTP request timeout in seconds
            delay_between_requests: Minimum delay between requests (per thread)
        """
        self.cache_dir = cache_dir              # caching for increased speed + checking we dont repeat articles
        self.max_threads = max_threads          # thread count var
        self.timeout = timeout                  # timeout to prevent crashing
        self.delay = delay_between_requests

        # create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def url_to_cache_key(self, url: str) -> str:
        # create filesystem safe cache key via url
        return hashlib.md5(url.encode()).hexdigest()


    def get_cached_url(self, url: str) -> Optional[str]:
        # check cache to see if article has already been extracted to prevent repetitions + speed
        cache_path = os.path.join(self.cache_dir, f"{self.url_to_cache_key(url)}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("content")
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def save_to_cache(self, url: str, content: Optional[str]):
        """Save article content to disk cache."""
        cache_path = os.path.join(self.cache_dir, f"{self.url_to_cache_key(url)}.json")
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "url": url,
                    "content": content,
                    "scraped_at": datetime.now().isoformat()
                }, f, ensure_ascii=False)
        except IOError as e:
            print(f"  Cache write error: {e}")

    def scrape_single_article(self, url: str) -> Optional[str]:
        """Fetch and extract article body text from a single URL.
        
        Args:
            url: Article URL to scrape
            
        Returns:
            Extracted article text, or None if extraction fails
        """
        if not trafilatura:
            print("Unable to find trafilatura installation.")
            return None

        # check cache first
        cached_url = self.get_cached_url(url)
        if cached_url is not None:
            return cached_url

        try:
            # download the page
            downloaded_article = trafilatura.fetch_url(url)
            if not downloaded_article:
                self.save_to_cache(url, None)
                return None

            # extract main article content
            content = trafilatura.extract(
                downloaded_article,
                include_comments=False,
                include_tables=False,
                no_fallback=False
            )

            # cache result (even if None, to avoid re-fetching failures)
            self.save_to_cache(url, content)

            time.sleep(self.delay)
            return content

        except Exception as e:
            self.save_to_cache(url, None)
            return None

    def scrape_articles(self, articles: List[NewsArticle],
                         show_progress: bool = True) -> List[NewsArticle]:
        """Scrape content for a list of articles using concurrent fetching.
        
        Modifies articles in-place by populating the `content` field.
        Articles that fail scraping retain content=None (title-only fallback).
        
        Args:
            articles: List of NewsArticle objects to scrape
            show_progress: Whether to print progress updates
            
        Returns:
            The same list of articles with content fields populated where possible
        """
        if not trafilatura:
            print("Unable to find trafilatura installation - skipping the content scraping")
            print("For Sentiment analysis will use titles only (valid for VADER)")
            return articles

        total = len(articles)
        if total == 0:
            return articles

        # check how many are already cached_url
        cache_hits = sum(1 for a in articles if self.get_cached_url(a.url) is not None)
        to_fetch = total - cache_hits

        if show_progress:
            print(f"\nContent scraping: {total} articles ({cache_hits} cached_url, {to_fetch} to fetch)")

        success_count = 0
        failed_count = 0

        # build URL  to -> article index mapping for concurrent processing
        url_to_indices: Dict[str, List[int]] = {}
        for i, article in enumerate(articles):
            if article.url not in url_to_indices:
                url_to_indices[article.url] = []
            url_to_indices[article.url].append(i)

        unique_urls = list(url_to_indices.keys())

        # used for concurrent fetching
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:              # threaadpoolexecutor must use max_workers for param, bug fix
            future_to_url = {
                executor.submit(self.scrape_single_article, url): url
                for url in unique_urls
            }

            completed = 0
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                completed += 1

                try:
                    content = future.result()

                    # assign content to all articles with this URL
                    for idx in url_to_indices[url]:
                        articles[idx].content = content

                    if content:
                        success_count += 1
                    else:
                        failed_count += 1

                except Exception as e:
                    failed_count += 1

                # progress update every 50 articles
                if show_progress and completed % 50 == 0:
                    print(f"  Progress: {completed}/{len(unique_urls)} URLs processed")

        if show_progress:
            print(f"  Scraping complete: {success_count} succeeded, {failed_count} failed")
            print(f"  Articles with content: {sum(1 for a in articles if a.content)}/{total}")

        return articles

    def clear_cache(self):
        # remove all cached article content from cache

        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            print("Successfully cleared Article cache")


    # function whcih returns the stats of the cache, for visibility of total, with content and failed caches
    def cache_stats(self) -> Dict[str, int]:
        
        if not os.path.exists(self.cache_dir):                  # if cache dir failed, return all 0's 
            return {"total_cached_url": 0, "with_content": 0, "failed": 0}

        total = 0                   # total cached_url articles
        with_content = 0            # no. articles with successful extraction of content
        failed = 0                  # no. articles with failed extraction / 0 content

        for file_name in os.listdir(self.cache_dir):
            if file_name.endswith('.json'):
                total += 1
                try:
                    filepath = os.path.join(self.cache_dir, file_name)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)                         # extract data from extracted article json
                        if data.get("content"):                     # if content key exists, success, else fail
                            with_content += 1
                        else:
                            failed += 1
                except (json.JSONDecodeError, IOError):
                    failed += 1

        return {"total_cached_url": total, "with_content": with_content, "failed": failed}




# Main function using an example url to test
if __name__ == "__main__":
    # quick test with reuters url to see if works
    scraper = ContentScraper()
    test_url = "https://finance.yahoo.com/news/clarus-clar-q4-2025-earnings-235825033.html"
    content = scraper.scrape_single_article(test_url)
    
    
    if content:
        print(f"Extracted {len(content)} chars from {test_url}")
        print(f"Content extracted: {content[:300]}...")
    else:
        print("No content was able to be extracted")

    print(f"\nCache statistics: {scraper.cache_stats()}")