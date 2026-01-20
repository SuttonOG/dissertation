from __future__ import annotations
import yfinance as yf                           # used here to get information for a ticker, e.g APPL, to then be used later to extarct 

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set


#Used to build a query pack from a user given ticker using yfinance and other methods
@dataclass(frozen = True)                   # freeze to be immutable -> shouldnt change
class QueryPack:
    ticker : str
    quote_type : str                # quote type = ETF, Crypto, equity etc
    company_name : str
    short_name : str
    sector : str
    industry : str
    country : str
    exchange : str
    business_summary : str

    name_aliases = List[str]                    # to be used for searching for e.g APPL/ Apple/ Apple Inc
    # disambiguators : str                        # used if a name is ambigous, e.g Apple 
    # news_API_querys : List[str]

    # to use if its a common stock, can be added to later - allows more detail than yfinance ticker extract
    Common_Stocks = {
        'AAPL': {
            'ticker': 'AAPL',
            'company_names': ['Apple', 'Apple Inc'],
            'ceo_executives': ['Tim Cook'],
            'products': ['iPhone', 'iPad', 'Mac', 'Apple Watch', 'Vision Pro', 'AirPods'],
            'exchange': 'NASDAQ',
            'industry_terms': ['consumer electronics', 'smartphone', 'tech giant'],
            'disambiguators': ['iPhone', 'Cupertino', 'NASDAQ', 'Tim Cook', 'iOS', 'App Store'],
            'is_ambiguous': True,
        },
        'NVDA': {
            'ticker': 'NVDA',
            'company_names': ['NVIDIA', 'Nvidia Corp'],
            'ceo_executives': ['Jensen Huang'],
            'products': ['GeForce', 'RTX', 'CUDA', 'A100', 'H100', 'DGX'],
            'exchange': 'NASDAQ',
            'industry_terms': ['GPU', 'graphics card', 'AI chips', 'semiconductor'],
            'disambiguators': ['Jensen Huang', 'NASDAQ', 'GPU', 'graphics'],
            'is_ambiguous': False,
        },
    }   

    # will use to sort queries for these keywords etcs
_financial_words = [
        "stock", "shares", "earnings", "forecast", "revenue", "profit", "NASDAQ", "NYSE", "SEC", "IPO", "dividend", "buyback", "acquisition"
    ]   

    # used for e.g Apple Inc or Microsoft corp
_company_endings = [
        "inc", "inc.", "corp", "corp.", "corporation", "ltd", "ltd.", "plc", "co", "co.", "group"
    ]
    

def remove_comapany_endings(name : str) -> str:

    if not name:
        return ""
        
        # else remove the company ending, e.g Apple Inc. to Apple

    total = re.split(r"\s+", name.strip())
    cleaned = []

    for part in total:
        part_normal = part.strip(",".lower())
        if part_normal in _company_endings:
            continue
        cleaned.append(part.strip(","))


    #Build a dict of information/build a pack to qwuery using yfinance ticker info
def collect_from_yf(ticker: str)  -> Optional[Dict]:
    
    #builds a querypack from yfinance ticker
    extracted_ticker = yf.Ticker(ticker)
    ticker_info : Dict = extracted_ticker.info or {}


    # checks to ensure it validated the ticker
    company_name = ticker_info.get("longName") or ticker_info.get("shortName") or ""
    quote_type = ticker_info.get("quoteType") or ""

    if not company_name and not company_name:
        raise ValueError(f"Was unable to get ticker information from yfinance for {ticker}")
    
    extracted_short_name = ticker_info.get("shortName") or company_name
    # remove company endings
    short_name = remove_comapany_endings(extracted_short_name)

    #extract rest of the ticker info for query pack
    sector = ticker_info.get("sector","")
    industry = ticker_info.get("industry", "")
    country = ticker_info.get("country","")
    exchange = ticker_info.get("exchange","") or ticker_info.get("fullExchangeName","")
    summary = ticker_info.get("longBusinessSummary", "") or ""

    # ADD Aliases for a company e.g AAPL - Apple Inc, Apple

    aliases : Set[str] = {ticker}
    if company_name:
        aliases.add(company_name)
    if short_name:
        aliases.add(short_name)
    # also add without the dot e.g Apple Inc
    if company_name:
        aliases.add(company_name.replace(".",""))           # remove dot 

    
    return QueryPack(
        ticker = ticker,
        quote_type = quote_type,
        sector = sector,
        short_name = short_name,
        industry=industry,
        country = country,
        exchange = exchange,
        business_summary=summary, 
        company_name = company_name
        # name_aliases = sorted(aliases)
    )

    
        # if not, its worked, continue building the query pack

    #         ticker : str
    # quote_type : str                # quote type = ETF, Crypto, equity etc
    # company_name : str
    # short_name : str
    # sector : str
    # industry : str
    # country : str
    # exchange : str
    # business_summary : str

    # name_aliases = List[str]                    # to be used for searching for e.g APPL/ Apple/ Apple Inc
    # disambiguators : str                        # used if a name is ambigous, e.g Apple 
    # news_API_querys : List[str]



            #format of a pack to build


    # @classmethod
    # def get_pack(cls, ticker : str) -> Dict[][]:




if __name__ == "__main__":
    test_pack = collect_from_yf("AAPL")
    print(asdict(test_pack))




    





