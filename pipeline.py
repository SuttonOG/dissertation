# Main pipeline for running the full script.
# takes a ticker, collects news, scores sentiment, clusters, and visualises

import argparse                     # parser for cmd line args 
import os
import time

from data_collection.gdelt_collect import GDELTCollector
from data_collection.rss_collector import RSSFeedCollector
from data_collection.content_scraper import ContentScraper
from data_collection.price_data import get_price_extracted_data
from processing.article_to_dataframe import convert_articles_to_dataframe
from processing.sentiment_vader import VaderScorer
from processing.feature_aggregate import build_feature_matrix
from analysis.clustering import run_clustering
from visualization.visualize_results import generate_all_charts



def run_pipeline(ticker: str = "NVDA", days_back: int = 2, max_records_per_day: int = 50, scrape: bool = False,  output_dir: str = "data", enable_rss : bool = True, cluster_method: str = "hdbscan", n_clusters: int = None, sentiment: str = "vader"):


        # Steps:
        # 1. Collect articles from GDELT (multi-day)
        # 2. Collect articles from RSS feeds  -- maybe optional for now, as theyre not very good
        # 3. Combine and deduplicate articles into a main DataFrame
        # 4. Optionally scrape article content 
        # 5. VADER Sentiment Scoring
        # 6. Fetch price data from yfinance
        # 7. Build feature matrix (aggregate daily sentiment + merge with prices)
        # 8. Clustering (HDBSCAN)
        # 9. Visualisations
        # 10. Save everything


    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    print("*" * 60)
    print("Running the pipeline...")
    print(f"\nTicker:    {ticker}")
    print(f"\nDays back: {days_back}")
    print(f"\nScraping = {'ON' if scrape else 'OFF'}")      

    # Step 1 - Run GDELT for ticker, extract articles
    print(f"\n ---- Step 1: GDELT Article Extraction for {ticker} ----")
    gdelt = GDELTCollector()
    gdelt_articles = gdelt.extract_multiple_days_using_ticker(
        ticker=ticker,
        days_backwards=days_back,
        max_records_per_day=max_records_per_day,
        delay_between_days=6.0              # keep as 6.0 or will be rate limited
    )
    print(f"GDELT Article Count: {len(gdelt_articles)}")


    # Step 2: RSS feed extractor - may comment out for now
    rss_articles = []
    if enable_rss:
        print(f"\n--- Step 2: RSS feed collection ---")
        rss = RSSFeedCollector()
        rss_articles = rss.collect_from_all_feeds(days_backwards=days_back)
        print(f"RSS articles: {len(rss_articles)}")
    else:
         print(f"\n RSS Feed Extraction disabled - Skipping....")

    # Step 3: Combine the extracted articles and deduplicate 
    print(f"\n --- Step 3: Combine extracted articles and deduplicate them ---")
    all_articles = gdelt_articles + rss_articles
    print(f"Total before duplicates removed: {len(all_articles)}")

    df = convert_articles_to_dataframe(all_articles)
    if df is None or df.empty:
        print("ERROR: No articles were collected. Exiting pipeline....")
        return

    # Step 4 - scrape content from articles extracted from GDELT (body etc)
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

    # Step 5: Sentiment scoring (VADER or FinBERT)
    print(f"\n--- Step 5: {sentiment.upper()} Sentiment Scoring ---")
    if sentiment == 'finbert':
        from processing.sentiment_finbert import FinBertScorer
        scorer = FinBertScorer()
        df = scorer.score_dataframe(df)
    else:
        vader = VaderScorer()
        df = vader.score_dataframe(df)
    
    # Step 6: Extract Price data for ticker (price_data)
    print(f"\n--- Step 6: Fetching price data ---")
    price_df = get_price_extracted_data(ticker=ticker, days_back=days_back)

    # Step 7: Build feature matrix (aggregate daily sentiment + merge with prices)
    print(f"\n--- Step 7: Building feature matrix ({sentiment.upper()}) ---")
    feature_matrix = build_feature_matrix(df, price_df, sentiment=sentiment)

    # Step 8: Clustering
    # adjust min_cluster_size based on how much data we have
    # for short runs (< 10 days) use smaller clusters so we actually get results
    min_cs = max(3, len(feature_matrix) // 10)      # roughly 10% of data points
    min_cs = min(min_cs, 10)                         # but cap it at 10

    print(f"\n--- Step 8: Clustering ({cluster_method.upper()}) ---")

    if cluster_method == 'hdbscan':
        print(f"  using min_cluster_size={min_cs} for {len(feature_matrix)} data points")
        feature_matrix, clusterer = run_clustering(
            feature_matrix,
            min_cluster_size=min_cs,
            min_samples=2,
            method='hdbscan',
            sentiment=sentiment,
        )
    elif cluster_method == 'kmeans':
        print(f"  n_clusters={'auto' if n_clusters is None else n_clusters}")
        feature_matrix, clusterer = run_clustering(
            feature_matrix,
            method='kmeans',
            n_clusters=n_clusters,
            sentiment=sentiment,
        )
    elif cluster_method == 'gmm':
        print(f"  n_components={'auto (BIC)' if n_clusters is None else n_clusters}")
        feature_matrix, clusterer = run_clustering(
            feature_matrix,
            method='gmm',
            n_clusters=n_clusters,
            sentiment=sentiment,
        )
    elif cluster_method == 'hmm':
        print(f"  n_states={'auto (BIC)' if n_clusters is None else n_clusters}")
        feature_matrix, clusterer = run_clustering(
            feature_matrix,
            method='hmm',
            n_clusters=n_clusters,
            sentiment=sentiment,
        )
    else:
        print(f"  Unknown method '{cluster_method}', falling back to HDBSCAN")
        feature_matrix, clusterer = run_clustering(
            feature_matrix,
            min_cluster_size=min_cs,
            min_samples=2,
            method='hdbscan',
            sentiment=sentiment,
        )

    # get cluster profiles for the summary
    profiles = clusterer.get_cluster_profiles(feature_matrix)

    # Step 8b: Statistical validation
    # tests whether the clusters actually have different return/volatility behaviour
    print(f"\n--- Step 8b: Statistical Validation ---")
    from analysis.statistical_validation import validate_clusters, save_validation_report
    validation_results = validate_clusters(feature_matrix)

    # Step 9: Generate visualisations
    print(f"\n--- Step 9: Generating visualisations ---")
    chart_dir = os.path.join(output_dir, "charts")
    generate_all_charts(
        feature_matrix=feature_matrix,
        articles_df=df,
        ticker=ticker,
        output_dir=chart_dir
    )

    # Step 10: Save everything
    print(f"\n--- Step 10: Saving outputs ---")

    articles_path = os.path.join(output_dir, f"articles_{ticker}_{days_back}d.csv")
    df.to_csv(articles_path, index=False)
    print(f"  Articles saved: {articles_path}")

    if price_df is not None:
        price_path = os.path.join(output_dir, f"prices_{ticker}_{days_back}d.csv")
        price_df.to_csv(price_path)
        print(f"  Prices saved: {price_path}")

    feature_path = os.path.join(output_dir, f"features_{ticker}_{days_back}d.csv")
    feature_matrix.to_csv(feature_path, index=False)
    print(f"  Feature matrix saved: {feature_path}")

    if not profiles.empty:
        profiles_path = os.path.join(output_dir, f"cluster_profiles_{ticker}_{days_back}d.csv")
        profiles.to_csv(profiles_path)
        print(f"  Cluster profiles saved: {profiles_path}")

    # save validation report
    if validation_results:
        validation_path = os.path.join(output_dir, f"validation_{ticker}_{days_back}d.csv")
        save_validation_report(validation_results, validation_path)

    # save HMM-specific outputs if we used HMM
    if cluster_method == 'hmm' and hasattr(clusterer, 'get_transition_matrix'):
        trans_matrix = clusterer.get_transition_matrix()
        if trans_matrix is not None:
            trans_path = os.path.join(output_dir, f"transition_matrix_{ticker}_{days_back}d.csv")
            trans_matrix.to_csv(trans_path)
            print(f"  Transition matrix saved: {trans_path}")

        durations = clusterer.get_expected_durations()
        if durations is not None:
            dur_path = os.path.join(output_dir, f"regime_durations_{ticker}_{days_back}d.csv")
            durations.to_csv(dur_path, index=False)
            print(f"  Regime durations saved: {dur_path}")

    # print summary to console
    time_taken = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"PIPELINE IS NOW     COMPLETE")
    print(f"  Ticker:         {ticker}")
    print(f"  Sentiment:      {sentiment.upper()}")
    print(f"  Cluster method: {cluster_method.upper()}")
    print(f"  Articles:       {len(df)}")

    if 'content' in df.columns:
        print(f"  With content:   {df['content'].notna().sum()}")


    print(f"  Price days:     {len(price_df) if price_df is not None else 0}")
    print(f"  Feature days:   {len(feature_matrix)}")

    n_clusters = feature_matrix[feature_matrix['cluster_label'] >= 0]['cluster_label'].nunique()

    print(f"  Clusters found: {n_clusters}")

    print(f"  Charts:         saved to {os.path.abspath(chart_dir)}")

    print(f"  Duration:       {time_taken:.1f} seconds")

    print(f"  Output dir:     {os.path.abspath(output_dir)}")
    
    print("=" * 60)



if __name__ == "__main__":

        # parser for passing function args when running pipeline
        parser = argparse.ArgumentParser(description="Financial news sentiment pipeline")
        parser.add_argument("--ticker", type=str, default="NVDA", help="Stock ticker (default: NVDA)")
        parser.add_argument("--days", type=int, default=2, help="Days of history (default: 2)")
        parser.add_argument("--max-per-day", type=int, default=50, help="Max GDELT articles per day (default: 50)")
        parser.add_argument("--scrape", action="store_true", help="Enable content scraping (off by default)")
        parser.add_argument("--output-dir", type=str, default="data", help="Output directory (default: data)")
        parser.add_argument("--rss", action="store_true", help="Enable RSS feed extraction")
        parser.add_argument("--method", type=str, default="hdbscan", choices=["hdbscan", "kmeans", "gmm", "hmm"],
                            help="Clustering method (default: hdbscan)")
        parser.add_argument("--k", type=int, default=None,
                            help="Number of clusters/components/states (default: auto-select)")
        parser.add_argument("--sentiment", type=str, default="vader", choices=["vader", "finbert"],
                            help="Sentiment scorer to use (default: vader)")

        args = parser.parse_args()

        run_pipeline(
            ticker=args.ticker,
            days_back=args.days,
            max_records_per_day=args.max_per_day,
            scrape=args.scrape,
            output_dir=args.output_dir,
            enable_rss=args.rss,
            cluster_method=args.method,
            n_clusters=args.k,
            sentiment=args.sentiment,
        )