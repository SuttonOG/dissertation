import pandas as pd
import numpy as np
from typing import Optional



def aggregate_aggregated_daily_sentiment_df_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    
    # Aggregate the sentiment scores of articles to aggregated_daily_sentiment_df features

    # Args:
    #    df: DataFrame with 'published_day' and 'vader_compound' columns with a column for each vader statistic for each day


    df = df.copy()                      # copy df so original isnt modified

    # ensure published_day is string for clean grouping
    df['published_day'] = pd.to_datetime(df['published_day']).dt.date                   # sometimes there is one that doesnt work correctly

    print(f"Aggregating sentiment for {df['published_day'].nunique()} unique days...")
    
    # group by date, calculate stats for the sentiment of articles for that day
    aggregated_daily_sentiment_df = df.groupby('published_day').agg(
        article_count=('vader_compound', 'count'),
        vader_mean=('vader_compound', 'mean'),
        vader_std=('vader_compound', 'std'),
        vader_median=('vader_compound', 'median'),
        vader_min=('vader_compound', 'min'),
        vader_max=('vader_compound', 'max'),
    ).reset_index()



    # compute sentiment ratios per day
    positive_counts = df[df['vader_compound'] > 0.05].groupby('published_day').size()
    negative_counts = df[df['vader_compound'] < -0.05].groupby('published_day').size()
    total_counts = df.groupby('published_day').size()

    aggregated_daily_sentiment_df = aggregated_daily_sentiment_df.set_index('published_day')            # set index to be date


    aggregated_daily_sentiment_df['positive_ratio'] = (positive_counts / total_counts).fillna(0)        
    aggregated_daily_sentiment_df['negative_ratio'] = (negative_counts / total_counts).fillna(0)
    aggregated_daily_sentiment_df = aggregated_daily_sentiment_df.reset_index()

    # Error handling - fill NaN strading_day (happens when only 1 article in a day) with 0
    aggregated_daily_sentiment_df['vader_std'] = aggregated_daily_sentiment_df['vader_std'].fillna(0)

    # sort by date
    aggregated_daily_sentiment_df = aggregated_daily_sentiment_df.sort_values('published_day').reset_index(drop=True)

    print(f"  Days with data: {len(aggregated_daily_sentiment_df)}")
    print(f"  Articles per day: min={aggregated_daily_sentiment_df['article_count'].min()}, "
          f"max={aggregated_daily_sentiment_df['article_count'].max()}, "
          f"mean={aggregated_daily_sentiment_df['article_count'].mean():.1f}")
    print(f"  Mean aggregated_daily_sentiment_df sentiment: {aggregated_daily_sentiment_df['vader_mean'].mean():.4f}")
    print(f"  Mean sentiment dispersion: {aggregated_daily_sentiment_df['vader_std'].mean():.4f}")

    return aggregated_daily_sentiment_df

# merge aggregated_daily_sentiment_df sentiment features with the price data extracted for the ticker
def merge_with_prices(aggregated_daily_sentiment_df_sentiment: pd.DataFrame,
                      price_df: pd.DataFrame) -> pd.DataFrame:
  
    
    #Error handling, for weekend or holiday news roll it forward to monday/ next trading day. Only trading days kept in final output

    # Arguments:
    # aggregated_daily_sentiment_df_sentiment - output from aggregate_aggregated_daily_sentiment_df_sentiment func
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

# Combine - aggregate aggregated_daily_sentiment_df sentiment, then combine on price data and return full df
def build_feature_matrix(articles_df: pd.DataFrame,
                         price_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:      #price data optional 


    # Arguments: articles_df = scored articles dataframe, price_df = optional price data dataframe

    # Returns: feature matrix with aggregated_daily_sentiment_df vader stats for each trading day + price data for those respective days.
    # Feature matrix to be used in clustering

    print(f"\n{'=' * 60}")
    print("Preparing Feature Matrix...")
    print(f"{'=' * 60}")

    # step 1: aggregate aggregated_daily_sentiment_df sentiment
    aggregated_daily_sentiment_df = aggregate_aggregated_daily_sentiment_df_sentiment(articles_df)

    # step 2: merge with prices if available
    if price_df is not None and not price_df.empty:
        print(f"\nMerging with price data...")
        feature_matrix = merge_with_prices(aggregated_daily_sentiment_df, price_df)
    else:
        print("\nNo price data provided - only returning sentiment features...")
        feature_matrix = aggregated_daily_sentiment_df


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

        # check if vader scores exist first 
        if 'vader_compound' not in articles_df.columns:
            print("No vader_compound column found - run VADER scoring first")
        else:
            price_df = None
            if os.path.exists(prices_csv):
                price_df = pd.read_csv(prices_csv, index_col=0, parse_dates=True)

            feature_matrix = build_feature_matrix(articles_df, price_df)            # build feature matrix with dataframes

            # save
            output_path = "data/feature_matrix.csv"
            feature_matrix.to_csv(output_path, index=False)
            print(f"\nFeature matrix saved to {output_path}")
    else:
        print(f"Unable to find any articles file found at {articles_csv}")
        print("If this occurs, run the pipeline first with  python pipeline.py --scrape")