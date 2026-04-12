import pandas as pd
import numpy as np
from typing import Optional


# maps each sentiment scorer to its compound column name in the articles dataframe
SENTIMENT_COMPOUND_COL = {
    'vader': 'vader_compound',
    'finbert': 'finbert_compound',
}


def aggregate_daily_sentiment(df: pd.DataFrame,
                              sentiment: str = 'vader') -> pd.DataFrame:
    
    # Aggregate article-level sentiment scores into daily features.
    #
    # Args:
    #    df: DataFrame with 'published_day' and a compound score column
    #    sentiment: which scorer to aggregate ('vader' or 'finbert')
    #
    # Returns:
    #    DataFrame with one row per day containing:
    #      article_count, {sentiment}_mean, {sentiment}_std, {sentiment}_median,
    #      {sentiment}_min, {sentiment}_max, positive_ratio, negative_ratio

    df = df.copy()                      # copy df so original isnt modified

    compound_col = SENTIMENT_COMPOUND_COL.get(sentiment)
    if compound_col is None:
        raise ValueError(f"Unknown sentiment scorer: '{sentiment}'. Use 'vader' or 'finbert'.")
    
    if compound_col not in df.columns:
        raise ValueError(f"Column '{compound_col}' not found in dataframe. "
                         f"Run {sentiment} scoring before aggregating.")

    # prefix for output columns so vader and finbert results dont collide
    prefix = sentiment

    # ensure published_day is date type for clean grouping
    df['published_day'] = pd.to_datetime(df['published_day']).dt.date

    print(f"Aggregating {sentiment.upper()} sentiment for {df['published_day'].nunique()} unique days...")
    
    # group by date, calculate stats for the sentiment of articles for that day
    agg = df.groupby('published_day').agg(
        article_count=(compound_col, 'count'),
        **{
            f'{prefix}_mean': (compound_col, 'mean'),
            f'{prefix}_std': (compound_col, 'std'),
            f'{prefix}_median': (compound_col, 'median'),
            f'{prefix}_min': (compound_col, 'min'),
            f'{prefix}_max': (compound_col, 'max'),
        }
    ).reset_index()

    # compute sentiment ratios per day (positive / negative article proportions)
    positive_counts = df[df[compound_col] > 0.05].groupby('published_day').size()
    negative_counts = df[df[compound_col] < -0.05].groupby('published_day').size()
    total_counts = df.groupby('published_day').size()

    agg = agg.set_index('published_day')

    agg['positive_ratio'] = (positive_counts / total_counts).fillna(0)        
    agg['negative_ratio'] = (negative_counts / total_counts).fillna(0)
    agg = agg.reset_index()

    # fill NaN std (happens when only 1 article in a day) with 0
    agg[f'{prefix}_std'] = agg[f'{prefix}_std'].fillna(0)

    # sort by date
    agg = agg.sort_values('published_day').reset_index(drop=True)

    print(f"  Days with data: {len(agg)}")
    print(f"  Articles per day: min={agg['article_count'].min()}, "
          f"max={agg['article_count'].max()}, "
          f"mean={agg['article_count'].mean():.1f}")
    print(f"  Mean daily sentiment: {agg[f'{prefix}_mean'].mean():.4f}")
    print(f"  Mean sentiment dispersion: {agg[f'{prefix}_std'].mean():.4f}")

    return agg

# merge aggregated_daily_sentiment_df sentiment features with the price data extracted for the ticker
def merge_with_prices(aggregated_daily_sentiment_df_sentiment: pd.DataFrame,
                      price_df: pd.DataFrame) -> pd.DataFrame:
  
    
    #Error handling, for weekend or holiday news roll it forward to monday/ next trading day. Only trading days kept in final output

    # Arguments:
    # aggregated_daily_sentiment_df_sentiment - output from aggregate_daily_sentiment func
    # price_df - the price data dataframe - contains 'date' column

    # Returns:
    # merged dataframe with sentiment features + price data.

    aggregated_daily_sentiment_df = aggregated_daily_sentiment_df_sentiment.copy()              # create copy of df so original isnt modified
    prices = price_df.copy()                    

    # ensure date columns are the same type
    aggregated_daily_sentiment_df['published_day'] = pd.to_datetime(aggregated_daily_sentiment_df['published_day'])         # convert to datetime

    if 'date' in prices.columns:
        prices['date'] = pd.to_datetime(prices['date'])
    else:
        prices['date'] = pd.to_datetime(prices.index)

    # merge on date
    # align weekend/holiday news to nearest trading day
    trading_dates = sorted(prices['date'].unique())
    
    def nearest_trading_day(news_date):

        # maps news date to nearest trading day e.g weekend goes to monday 
       
        for trading_day in trading_dates:
            if trading_day >= news_date:
                return trading_day
        # if no future trading day found, use the most recent past one
        for trading_day in reversed(trading_dates):
            if trading_day <= news_date:
                return trading_day
        return None

    aggregated_daily_sentiment_df['trading_day'] = aggregated_daily_sentiment_df['published_day'].apply(nearest_trading_day)

    # merge on aligned trading day
    merged = pd.merge(
        aggregated_daily_sentiment_df,
        prices[['date', 'daily_return', 'log_return',
                'realised_volatility_5d', 'realised_volatility_20d']],
        left_on='trading_day',
        right_on='date',
        how='inner'
    )
    # drop redundant date column
    if 'date' in merged.columns:
        merged = merged.drop(columns=['date'])

    # drop rows with missing returns
    before = len(merged)
    merged = merged.dropna(subset=['daily_return'])
    dropped = before - len(merged)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing return data")

    print(f"  Merged feature matrix: {len(merged)} trading days")
    print(f"  Columns: {list(merged.columns)}")

    return merged

# Combine - aggregate daily sentiment, then combine on price data and return full df
def build_feature_matrix(articles_df: pd.DataFrame,
                         price_df: Optional[pd.DataFrame] = None,
                         sentiment: str = 'vader') -> pd.DataFrame:


    # Arguments: articles_df = scored articles dataframe, price_df = optional price data dataframe
    #            sentiment = which scorer was used ('vader' or 'finbert')

    # Returns: feature matrix with daily sentiment stats for each trading day + price data for those respective days.
    # Feature matrix to be used in clustering

    print(f"\n{'=' * 60}")
    print(f"Preparing Feature Matrix ({sentiment.upper()})...")
    print(f"{'=' * 60}")

    # step 1: aggregate daily sentiment using the chosen scorer
    daily_sentiment = aggregate_daily_sentiment(articles_df, sentiment=sentiment)

    # step 2: merge with prices if available
    if price_df is not None and not price_df.empty:
        print(f"\nMerging with price data...")
        feature_matrix = merge_with_prices(daily_sentiment, price_df)
    else:
        print("\nNo price data provided - only returning sentiment features...")
        feature_matrix = daily_sentiment


    # print feature matrix stats for error handling
    print(f"\nFeature matrix summary:")
    print(feature_matrix.describe().round(4))

    return feature_matrix

# testing block
if __name__ == "__main__":
    import os

    # test with existing CSV files
    articles_csv = "data/articles_NVDA_2d.csv"
    prices_csv = "data/prices_NVDA_2d.csv"

    if os.path.exists(articles_csv):
        articles_df = pd.read_csv(articles_csv)

        price_df = None
        if os.path.exists(prices_csv):
            price_df = pd.read_csv(prices_csv, index_col=0, parse_dates=True)

        # test VADER aggregation
        if 'vader_compound' in articles_df.columns:
            feature_matrix = build_feature_matrix(articles_df, price_df, sentiment='vader')

            output_path = "data/feature_matrix.csv"
            feature_matrix.to_csv(output_path, index=False)
            print(f"\nFeature matrix saved to {output_path}")
        else:
            print("No vader_compound column found - run VADER scoring first")

        # test FinBERT aggregation if available
        if 'finbert_compound' in articles_df.columns:
            feature_matrix_fb = build_feature_matrix(articles_df, price_df, sentiment='finbert')

            output_path_fb = "data/feature_matrix_finbert.csv"
            feature_matrix_fb.to_csv(output_path_fb, index=False)
            print(f"\nFinBERT feature matrix saved to {output_path_fb}")
    else:
        print(f"Unable to find any articles file found at {articles_csv}")
        print("If this occurs, run the pipeline first with  python pipeline.py --scrape")