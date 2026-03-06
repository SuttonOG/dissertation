# Main pipeline for running the full script.

import argparse                     # parser for cmd line args 
import os
import time

from data_collection.gdelt_collect import GDELTCollector
from data_collection.rss_collector import RSSFeedCollector
from data_collection.content_scraper import ContentScraper
from data_collection.price_data import get_price_extracted_data
from processing.article_to_dataframe import convert_articles_to_dataframe




def run_pipeline(ticker: str = "NVDA", days_back: int = 2, max_records_per_day: int = 50, scrape: bool = False,  output_dir: str = "data"):


        # Steps:
        # 1. Collect articles from GDELT (multi-day)
        # 2. Collect articles from RSS feeds  -- maybe optional for now, as theyre not very good
        # 3. Combine and deduplicate articles into a main DataFrame
        # 4. Optionally scrape article content 
        # 5. Fetch price data from yfinance
        # 6. Save everything


    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    print("*" * 60)
    print("Running the pipeline...")
    print(f"\nTicker:    {ticker}")
    print(f"\nDays back: {days_back}")
    print(f"\nScraping = {'ON' if scrape else 'OFF'}")      

    # Step 1 - Run GDELT for ticker, extract articles
    print(f"\n ---- Step 1 GDELT Article Extraction for {ticker} ----")
    gdelt = GDELTCollector()
    gdelt_articles = gdelt.extract_multiple_days_using_ticker(
        ticker=ticker,
        days_backwards=days_back,
        max_records_per_day=max_records_per_day,
        delay_between_days=6.0              # keep as 6.0 or will be rate limited
    )
    print(f"GDELT Article Count: {len(gdelt_articles)}")


    