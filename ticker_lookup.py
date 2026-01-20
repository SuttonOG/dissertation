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

    name_aliases : List[str]                    # to be used for searching for e.g APPL/ Apple/ Apple Inc
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

    # will use to sort main_terms for these keywords etcs
_financial_words = [
        "stock", "shares", "earnings", "forecast", "revenue", "profit", "NASDAQ", "NYSE", "SEC", "IPO", "dividend", "buyback", "acquisition"
    ]   

    # used for e.g Apple Inc or Microsoft corp
_company_endings = [
        "inc", "inc.", "corp", "corp.", "corporation", "ltd", "ltd.", "plc", "co", "co.", "group"
    ]
    

def remove_company_endings(name : str) -> str:

    if not name:
        return ""
        
        # else remove the company ending, e.g Apple Inc. to Apple

    words = re.split(r"\s+", name.strip())
    cleaned = []

    for part in words:
        part_normal = part.strip(",").lower()
        if part_normal in _company_endings:
            continue
        cleaned.append(part.strip(","))

    return " ".join(cleaned) if cleaned else name



#Build a dict of information/build a pack to qwuery using yfinance ticker info

def collect_from_yf(ticker: str)  -> Optional[Dict]:
    
    #builds a querypack from yfinance ticker
    extracted_ticker = yf.Ticker(ticker)
    ticker_info : Dict = extracted_ticker.info or {}


    # checks to ensure it validated the ticker
    company_name = ticker_info.get("longName") or ticker_info.get("shortName") or ""
    quote_type = ticker_info.get("quoteType") or ""

    if not company_name and not quote_type:
        raise ValueError(f"Was unable to get ticker information from yfinance for {ticker}")
    
    extracted_short_name = ticker_info.get("shortName") or company_name
    # remove company endings
    short_name = remove_company_endings(extracted_short_name)

    #extract rest of the ticker info for query pack
    sector = ticker_info.get("sector","")
    industry = ticker_info.get("industry", "")
    country = ticker_info.get("country","")
    exchange = ticker_info.get("exchange","") or ticker_info.get("fullExchangeName","")
    summary = ticker_info.get("longBusinessSummary", "") or ""

    # ADD Aliases for a company e.g AAPL - Apple Inc, Apple

    aliases : Set[str] = {ticker.upper()}
    if company_name:
        aliases.add(company_name)
        aliases.add(company_name.replace(".",""))
    if short_name and short_name != company_name:
        aliases.add(short_name)
    # also add without the dot e.g Apple Inc
          # remove dot 
    sorted_aliases = tuple(sorted(aliases))

    
    return QueryPack(
        ticker = ticker,
        quote_type = quote_type,
        sector = sector,
        short_name = short_name,
        industry=industry,
        country = country,
        exchange = exchange,
        business_summary=summary, 
        company_name = company_name,
        name_aliases = sorted_aliases
    )

    


def build_query_from_pack(query_pack: Dict[str,any]) -> str:
    # buiild a basic query from the query pack for more specific article extraction
    
    ticker = (query_pack.get("ticker") or "").strip()
    company = (query_pack.get("company_name") or "").strip()
    short_name = (query_pack.get("short_name") or "").strip()
    industry = (query_pack.get("industry") or "").strip()
    sector = (query_pack.get("sector") or "").strip()

    # OR logic for querying
    main_terms = []
    if company:
        main_terms.append(f"\"{company}\"")
    if short_name and short_name.lower() != company.lower():            # if short name different to company name also add
        main_terms.append(f"\"{short_name}\"")
    if ticker:
        main_terms.append(ticker)
    
    # if common ticker and creates noise, add additional info 
    secondary_terms = []
    if industry:
        secondary_terms.append(f"\"{industry}\"")
    if sector:
        secondary_terms.append(f"\"{sector}\"")

    # Query format = (company OR short_name OR ticker) AND (sector OR industry)
    if secondary_terms and main_terms:
        return f"({' OR '.join(main_terms)}) AND ({' OR '.join(secondary_terms)})"
    if main_terms:
        return " OR ".join(main_terms)

    # if pack hasnt extracted sufficient information - FALLBACk

    return company or ticker or ""


def build_query_for_gdelt(query_pack : Dict[str,any]) -> str:
    
    #Extract info from query pack to build GDELT query
    ticker = (query_pack.get("ticker") or "").strip().upper()
    company = (query_pack.get("company_name") or "").strip()
    short_name = (query_pack.get("short_name") or "").strip()
    aliases = query_pack.get("name_aliases", [])

    search_terms = set()

    if ticker:
        search_terms.add(ticker)
    if company:
        search_terms.add(f'"{company}"')
    if short_name and short_name.lower() != company.lower():
        search_terms.add(f'"{short_name}"')

    # add aliases also
    for alias in aliases:
        alias = alias.strip()  # remove whitespace
        if alias and alias.upper() != ticker:
            if ' ' in alias:
                search_terms.add(f'"{alias}"')
            else:
                search_terms.add(alias)
    
    if not search_terms:
        return ticker or company or ""
    
    return f"({' OR '.join(sorted(search_terms))})"


if __name__ == "__main__":

    # test for gdelt query building
    test_pack = collect_from_yf("AAPL")
    print(asdict(test_pack))

    pack_asdict = asdict(test_pack)
    query = build_query_from_pack(pack_asdict)

    print(f"Query generated: {query}")


    gdelt_query = build_query_for_gdelt(pack_asdict)
    print(f"\nGDELT Query: {gdelt_query}")




    





