import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional


# function to extract financial data from yfinance for a specific ticker and export to dataframe
def get_price_extracted_data(ticker: str = "NVDA",
                     days_back: int = 30,                   # default to 60 days 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    
    """
    extarcts the daily OHLCV price extracted_data from Yahoo Finance.
        ticker: yfinance (stock) ticker for the company
        days_back: No. days to look back 
        start_date: start date as 'YYYY-MM-DD'
        end_date: end date as 'YYYY-MM-DD' - default = today
        
    returns:
    extracted_dataFrame with columns: Open, High, Low, Close, Volume, daily_return, 
    log_return, realised_volatility_5d, realised_volatility_20d
    Returns None if download fails.
    """

    # date range
    if end_date:
        end_dt = pd.to_datetime(end_date)
    else:
        end_dt = datetime.now()

    if start_date:
        start_dt = pd.to_datetime(start_date)
    else:
        # add buffer days to account for weekends/holidays - UPDATED so on a weekend it still catches tarding days
        start_dt = end_dt - timedelta(days=max(days_back + 10, 14))

    print(f"Extracting price extracted_data for {ticker}: {start_dt.date()} to {end_dt.date()}")

    # extract the data from yfinance via download
    try:
        extracted_data = yf.download(
            ticker,
            start=start_dt.strftime('%Y-%m-%d'),
            end=end_dt.strftime('%Y-%m-%d'),
            progress=False
        )

        if extracted_data.empty:
            print(f"Unable to extract data for {ticker}")
            return None

        # flatten multi-level columns if present, added as yfinance adds these sometimes annoyingly
        if isinstance(extracted_data.columns, pd.MultiIndex):
            extracted_data.columns = extracted_data.columns.get_level_values(0)

        # compute daily returns for the stock
        extracted_data['daily_return'] = extracted_data['Close'].pct_change()

        # compute log returns for financial analysis
        extracted_data['log_return'] = np.log(extracted_data['Close'] / extracted_data['Close'].shift(1))

        # compute realised volatility (rolling std of returns)
        extracted_data['realised_volatility_5d'] = extracted_data['daily_return'].rolling(window=5).std()
        extracted_data['realised_volatility_20d'] = extracted_data['daily_return'].rolling(window=20).std()

        # add a date column from the index for easier merging later
        extracted_data['date'] = extracted_data.index.date

        # trim to the actual requested period (remove buffer days)
        if not start_date:
            actual_start = end_dt - timedelta(days=days_back)
            extracted_data = extracted_data[extracted_data.index >= pd.Timestamp(actual_start)]

        # drop the first row (NaN return)
        extracted_data = extracted_data.dropna(subset=['daily_return'])

        print(f"Retrieved {len(extracted_data)} trading days for {ticker}")
        print(f"  Date range: {extracted_data.index.min().date()} to {extracted_data.index.max().date()}")
        print(f"  Mean daily return: {extracted_data['daily_return'].mean():.4f}")
        print(f"  Return std dev: {extracted_data['daily_return'].std():.4f}")

        return extracted_data

    except Exception as e:
        print(f"Error fetching price extracted_data for {ticker}: {e}")
        return None
    

# maiun block for testing it works correctly
if __name__ == "__main__":
    df = get_price_extracted_data(ticker="NVDA", days_back=30)                  # test with nvidia for now
    

    # print the values found
    if df is not None:
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nLast 5 rows:")
        print(df[['Close', 'daily_return', 'realised_volatility_5d']].tail())
    else:
        print("No data retrieved")