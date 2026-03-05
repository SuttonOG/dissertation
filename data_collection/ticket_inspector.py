import yfinance as yf
import json
from typing import List, Optional, Dict


# This file will be used to extract ticker information from yfinance to see what we can use in the query.


def inspect_ticker(ticker : str):

    # Using yfinance, extract metadata for a ticket

    ticker_object = yf.Ticker(ticker)



    try:
        info = ticker_object.info


    except Exception as e:
        print(f"Failed to retrieve ticker info from {ticker}")
        return
    

    if not info:
        print(f"Retrieved but No data found for {ticker}")
        return # 
    
    # If info returned

    print("\n MetaData found for Ticker : {ticker}")
    print("-" * 60)

    key_fields = [
        "symbol",
        "shortName",
        "longName",
        "quoteType",
        "exchange",
        "currency",
        "sector",
        "industry",
        "country",
        "website",
        "longBusinessSummary"
    ]




    

    filtered = {k: info.get(k) for k in key_fields if k in info}
    print(json.dumps(filtered, indent=2))

    print("\nAvailable metadata keys:")
    print("-" * 60)
    print(sorted(info.keys()))

if __name__ == "__main__":
    inspect_ticker("AAPL")



    

    