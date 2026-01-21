import requests
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union,Any
import time
import re
import urllib.parse
import json
from rss_collector import NewsArticle
from ticker_lookup import QueryPack, collect_from_yf, build_query_from_pack
import requests
# this will be how articles are extracted using GDELT




class GDELTCollector:

    # GDELT provides access up to 3 months of data

    GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    MAX_RECORDS = 250                   # GDELT only allows a maximum of 250 records, can iterate requests for more i guess

    TIMEOUT = 30                        # DEFault request timeout allowed

    REQUEST_DELAY = 1.0                 # TIME BETWEEN requests, limits API intensity



    def __init__(self, timeout : int = 30):
        
        self.timeout = timeout
        self.last_request_time = 0



    def restrict_rate_limit(self):
        time_elapsed = time.time() - self.last_request_time
        if time_elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - time_elapsed)
        self.last_request_time = time.time()

    def parse_gdelt_datetime(self, date_string : str) -> datetime:
         # gdelt returns dates in wrong format: YYYYMMDDHHMMSS

        if not date_string:
            return datetime.now()
         
         #clean any bad characters
        date_string = re.sub(r'[^0-9]', '', date_string)

        # convert the extracted articles date to the correct format
        try:
            if len(date_string) >= 14:
                return datetime.strptime(date_string[:14],'%Y%m%d%H%M%S')
            elif len(date_string) >= 8:
                return datetime.strptime(date_string[:8], '%Y%m%d')
            else:
                return datetime.now()
        except ValueError as e:
            return datetime.now()
        
        
                

    def format_time_for_gdelt(self, dt : datetime) -> datetime:
        return dt.strftime('%Y%m%d%H%M%S')
    

    def build_query_for_gdelt(self,query_pack : Dict[str,Any]) -> str:
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
        
        # used to filter out rubbish news articles - for example gaming PC's for nvidia - can later build
        financial_terms = [
            "stock","shares","earnings","revenue","guidance","forecast","results","shareholders","profit","profits","all time high","ATH","bullish","bearish","investors"
        ]
        
        core_query = " OR ".join(search_terms)
        context_query = " OR ".join(financial_terms)
        
        return f"(({core_query}) AND ({context_query}))"
    
    # extracts articles after the gdelt query pack has been made
    def extract_articles(self, query: str, timespan: str = None, start_datetime: datetime = None, 
                        end_datetime : datetime = None, max_records: int = 250, sort: str = 'datedesc',
                        source_lang : str = None, source_country : str = None, domain: str = None) -> List[NewsArticle]:

            # example GDELT parameters for building
            # query: GDELT query string (supports boolean operators)
            # timespan: Time range like "1d", "1w", "3months"
            # start_datetime: Precise start time (alternative to timespan)
            # end_datetime: Precise end time (alternative to timespan)
            # max_records: Maximum articles to return (up to 250)
            # sort: Sort order - 'datedesc', 'dateasc', 'tonedesc', 'toneasc', 'hybridrel'
            # source_lang: Filter by source language (e.g., 'english', 'spanish')
            # source_country: Filter by source country (e.g., 'US', 'UK')
            # domain: Filter by domain (e.g., 'reuters.com')
            self.restrict_rate_limit()

            #build full query
            combined_query = query

            # these must be added to the query for GDELT
            if source_country:
                combined_query += f" sourcecountry:{source_country}"            # add specific country if desired
            if source_lang:
                combined_query += f" sourcelang:{source_lang}"               # add specific language if desired
            if domain:
                combined_query += f" domain:{domain}"                           # add domain

            # build parameters to pass to url
            parameters = {
                'query': combined_query,
                'mode' : 'artlist',                                          # mode = artlist generates list of news articles that match query - what we need
                'format' : 'json',
                'sort' : sort,                                               # set to datedesc = descending order (most recent articles first) by default
                'maxrecords': min(max_records, self.MAX_RECORDS)             # use specified amount if given, default to max extraction of 250
            }

            # add time parameters, but requires conversion
            if start_datetime and end_datetime:
                parameters['startdatetime'] = self.format_time_for_gdelt(start_datetime)
                parameters['enddatetime']= self.format_time_for_gdelt(end_datetime)
            elif timespan:
                # if timespan given, just pass it
                parameters['timespan'] = timespan

            extracted_articles = []

            try:
                print(f"\n Beginning GDELT Query for {combined_query}")
                response = requests.get(self.GDELT_URL, params=parameters, timeout=self.TIMEOUT)        # url, parameters, 1ms timeout 
                response.raise_for_status()
                
                response_data = response.json()                         # convert to json for extracting articles

                # GDELT returns articles in ['articles'] key
                if 'articles' in response_data:
                    # if articles exists, convert to NewsArticle Objects
                    for article in response_data['articles']:
                        try:
                            new_article = NewsArticle(
                                title=article.get('title', ''),
                                url=article.get('url',''),
                                published_date=self.parse_gdelt_datetime(article.get('seendate', '')),
                                source = f"gdelt_{article.get('domain','unknown')}",
                                content = None,
                            )
                            extracted_articles.append(new_article)

                        except Exception as e:
                            print(f"Error extracting articles. Error code : {e}")
            except requests.exceptions.Timeout:
                print(f"The GDELT Request timed out")
            except requests.exceptions.RequestException as e:                   # exception if the request failed
                print(f"Failed to request {self.GDELT_URL}")
            except ValueError as e:
                print(f"Error in parsing request response : {e}")
            
            return extracted_articles                           # if it doesnt fail, return all of the extracted files


        
    # way to extract by passing ticker
    def extract_using_ticker(self, ticker : str, days_back: int = 7, max_records: int = 250, source_language: str = 'english', sort: str = 'datedesc') -> List[NewsArticle]:

        # first build query 
        try:
            query_pack = collect_from_yf(ticker)                 # should return querypack 
            print(f"Query pack built for {ticker} : {query_pack.company_name}")
        except Exception as e:
            print(f"Unable to build query to extract for {ticker}, error code: {e}")
            # will add fallback later to search normally

        # now we have query pack, search for articles using it
        gdelt_query = self.build_query_for_gdelt(asdict(query_pack))

        return self.extract_articles(
                                    query = gdelt_query,
                                    max_records=max_records,
                                    source_lang = source_language,
                                    sort = sort,
                                    )
    
    
    
# used only for testing purposes
def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(...)



if __name__ == "__main__":

    # here for testing purposes

    gdelt_collector = GDELTCollector()
    articles = gdelt_collector.extract_using_ticker('NVDA', max_records=50)

    with open('gdelt_test','w') as f:
        json.dump([asdict(a) for a in articles], f, indent=2,default=_json_default)
        
