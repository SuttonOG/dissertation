# Main pipeline for running the full script.

import argparse                     # parser for cmd line args 
import os
import time

from data_collection.gdelt_collect import GDELTCollector
from data_collection.rss_collector import RSSFeedCollector
from data_collection.content_scraper import ContentScraper
from data_collection.price_data import get_price_extracted_data
from processing.article_to_dataframe import convert_articles_to_dataframe




def run_pipeline(ticker: str = "NVDA", days_back: int = 2, max_records_per_day: int = 50, scrape: bool = False,  output_dir: str = "data", enable_rss : bool = True):


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


    #Step 2: RSS feed extarctor - may comment out for now
    if enable_rss:
        print(f"\n--- Step 2: RSS feed collection ---")
        rss = RSSFeedCollector()
        rss_articles = rss.collect_from_all_feeds(days_backwards=days_back)
        print(f"RSS articles: {len(rss_articles)}")
    else:
         print(f"\n RSS Feed Extraction disabled - Skipping....")

    #Step 3: Combine the extracted articles and deduplicate 
    print(f"\n --- Step 3: Combine extracted articles and deduplicate them ---")
    all_articles = gdelt_articles + rss_articles
    print(f"Total before duplicates removed: {len(all_articles)}")

    df = convert_articles_to_dataframe(all_articles)
    if df is None or df.empty:
        print("ERROR: No articles collected. Exiting pipeline....")
        return

    #Step 4 - scrape content from articles extracted from GDELT (body etc)
    if scrape:
        print(f"\n--- Step 4: Content scraping articles ---")
        scraper = ContentScraper(
            cache_dir=os.path.join(output_dir, "cache", "articles"),
            max_threads=10
        )
        scraper.scrape_articles(all_articles, show_progress=True)

        # rebuild DataFrame with content populated
        df = convert_articles_to_dataframe(all_articles)
        content_count = df['content'].notna().sum() if 'content' in df.columns else 0
        print(f"Articles found with body content: {content_count}/{len(df)}")
    else:
        print(f"\n--- Step 4: Scraping SKIPPED ---")           # skip if scraping set to 0


    
    # Step 5: Extract Price data for ticker (price_data)
    print(f"\n--- Step 5: Fetching price data ---")
    price_df = get_price_extracted_data(ticker=ticker, days_back=days_back)

    # Step 6: Save 
    print(f"\n--- Step 6: Saving outputs ---")

    articles_path = os.path.join(output_dir, f"articles_{ticker}_{days_back}d.csv")
    df.to_csv(articles_path, index=False)
    print(f"Articles saved: {articles_path}")


    if price_df is not None:
        price_path = os.path.join(output_dir, f"prices_{ticker}_{days_back}d.csv")
        price_df.to_csv(price_path)
        print(f"Prices saved: {price_path}")


    # print summary to console

    time_taken = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Articles:       {len(df)}")
    if 'content' in df.columns:
        print(f"  With content:   {df['content'].notna().sum()}")
    print(f"  Price days:     {len(price_df) if price_df is not None else 0}")
    print(f"  Duration:   {time_taken:.1f} seconds")
    print(f"  Output directory:     {os.path.abspath(output_dir)}")
    print("=" * 60)



if __name__ == "__main__":

        # parser for passing function args when running pipeline
        parser = argparse.ArgumentParser(description="Financial news data collection pipeline")
        parser.add_argument("--ticker", type=str, default="NVDA", help="Stock ticker (default: NVDA)")
        parser.add_argument("--days", type=int, default=2, help="Days of history (default: 2)")
        parser.add_argument("--max-per-day", type=int, default=50, help="Max GDELT articles per day (default: 50)")
        parser.add_argument("--scrape", action="store_true", help="Enable content scraping (off by default)")
        parser.add_argument("--output-dir", type=str, default="data", help="Output directory (default: data)")
        parser.add_argument("--rss", action="store_true", help="Enable RSS feed extraction (default = True)")

        args = parser.parse_args()

        run_pipeline(
            ticker=args.ticker,
            days_back=args.days,
            max_records_per_day=args.max_per_day,
            scrape=args.scrape,
            output_dir=args.output_dir
        )
