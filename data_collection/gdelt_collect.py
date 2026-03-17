import requests
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union,Any
import time
import re
import urllib.parse
import json


# this will be how articles are extracted using GDELT

from data_collection.models import NewsArticle
from data_collection.ticker_lookup import QueryPack, collect_from_yf, build_query_for_gdelt
from processing.article_to_dataframe import convert_articles_to_dataframe


class GDELTCollector:

    # GDELT provides access up to 3 months of data

    GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    MAX_RECORDS = 250                   # GDELT only allows a maximum of 250 records, can iterate requests for more i guess

    TIMEOUT = 30                        # DEFault request timeout allowed

    REQUEST_DELAY = 5.0                 # TIME BETWEEN requests, limits API intensity

    MULTIPLE_DAYS_REQUEST_DELAY = 6.0     # gdelt API recommends at least 5 seconds inbetween API calls to rate limit

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
    

    # extracts articles after the gdelt query pack has been made
    def extract_articles(self, query: str, timespan: str = None, start_datetime: datetime = None, 
                        end_datetime : datetime = None, max_records: int = 250, sort: str = 'datedesc',
                        source_lang : str = None, source_country : str = None, domain: str = None) -> List[NewsArticle]:

            # example GDELT parameters for building
            # query: GDELT query string (supports boolean operators)
            # timespan: Time range like "1d", "1w", "3months"
            # start_datetime: Precise start time (alternative to timespan)
            # end_datetime: Precise end time (alternative to timespan)
            # max_records: max articles to return (up to 250)
            # sort: Sort order - 'datedesc', 'dateasc', 'tonedesc', 'toneasc', 'hybridrel'
            # source_lang: filter by source language (e.g., 'english', 'spanish')
            # source_country: filter by source country (e.g., 'US', 'UK')
            # domain: filter by domain (e.g., 'reuters.com')
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
            else:
                parameters['timespan'] = '7d'  # added default timespan

            extracted_articles = []

            try:
                print(f"\n Beginning GDELT Query for {combined_query}")
                response = requests.get(self.GDELT_URL, params=parameters, timeout=self.TIMEOUT)

                # handle rate limiting with retries
                retries = 0
                while response.status_code == 429 and retries < 3:
                    retries += 1
                    wait_time = 10 * retries
                    print(f"  Rate limited by GDELT. Retry {retries}/3, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    response = requests.get(self.GDELT_URL, params=parameters, timeout=self.TIMEOUT)

                response.raise_for_status()

                # error handling - check for empty response
                response_text = response.text.strip()
                if not response_text:
                    print("GDELT returned 0 articles/ no articles found")
                    return []           

                # error handling - check if response is actually JSON
                if not response_text.startswith('{'):
                    print("GDELT didnt return a JSON response")
                    return []               # stop execution if non-json response

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
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else "unknown"
                body = e.response.text[:500] if e.response else "no response body"
                print(f"HTTP Error {status_code} when using GDELT")
                print(f"Response body: {body}")
                print(f"Request URL: {e.response.url if e.response else 'unknown'}")
                print(body[:1000])
            except ValueError as e:
                print(f"Error in parsing request response : {e}")
            except Exception as e:
                print(f"Unexpected error: {type(e).__name__}: {e}")     # place Exception after valueerorr 

            
            
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
        gdelt_query = build_query_for_gdelt(asdict(query_pack))

        time_span = f'{days_back}d'
        return self.extract_articles(
                                    query = gdelt_query,
                                    max_records=max_records,
                                    source_lang = source_language,
                                    sort = sort,
                                    timespan = time_span
                                    )
    

    # method for extracting multiple days of articles
    def extract_articles_multiple_days(self,
                                       query : str,
                                       days_backwards : int = 30,               # set to do 1 month of historical data by default
                                       max_records_per_day: int = 50,
                                       delay_between_days: float = 6.0,
                                       sort : str = 'datedesc',
                                       source_lang : str = 'english',
                                       source_country : str = None,
                                       domain : str = None,
                                       ) -> List[NewsArticle]:
        
    # extract articles for multiple days by making a seperate API call for each day

        all_articles = [] 
        print(f"Benginning article extraction for {days_backwards} days of data.")
        print(f"Delay between requests: {delay_between_days}")


        for day_offset in range(days_backwards):
            print(f"Benginning article extraction for {days_backwards} days of data.")
            day_start = datetime.now().replace(hour = 0, minute=0,second=0,microsecond=0) - timedelta(days = day_offset)        # set start day to today midnight, then yesterday midnght etc until days_offset is 30 or days_backwards
            day_end = day_start.replace(hour = 23, minute=59, second=59)

            #format date for display
            date_string = day_start.strftime('%Y-%m-%d')
            print(f"\n Extracting for day {day_offset + 1}/{days_backwards}: {date_string}")

            # extract the articles for this specific day
            daily_articles = self.extract_articles(
                query=query,
                start_datetime=day_start,
                end_datetime=day_end,
                max_records=max_records_per_day,
                sort=sort,
                source_lang=source_lang,
                source_country=source_country,
                domain=domain
            )

            # after extracting 
            print(f"Number of articles for {date_string}: {len(daily_articles)}")
            #extend total articles with new articles
            all_articles.extend(daily_articles)

            # delay between requests as long as it isnt the last iteration
            if day_offset < days_backwards - 1:                 # add sleep unless its last iteration

                print(f"Waiting {delay_between_days} seconds before next call...")
                time.sleep(delay_between_days)              # 6 second delay for API rate limiting


        # after loop, return all articles

        print(f"\nCompleted Extraction of {days_backwards} days worth of articles. Total articles extracted: {len(all_articles)}")

        # return combined list of articles
        return all_articles                 # returns a list of news article objects



    def extract_multiple_days_using_ticker(
        self,
        ticker: str,
        days_backwards: int = 30,
        max_records_per_day: int = 50,
        delay_between_days: float = 6.0,
        source_language: str = 'english',
        sort: str = 'datedesc'
    ) -> List[NewsArticle]:
        
        try:
            # buiild queyry pack for ticker
            query_pack = collect_from_yf(ticker)
            print(f"Query pack built for {ticker}: {query_pack.company_name}")
        except Exception as e:
            print(f"Unable to build query for {ticker}, error: {e}")
            return []
    
    # build gdelt query from pack
        gdelt_query = build_query_for_gdelt(asdict(query_pack))
    
    # Use multi-day extraction - pass to multi-day extract method
        return self.extract_articles_multiple_days(
            query=gdelt_query,
            days_backwards=days_backwards,
            max_records_per_day=max_records_per_day,
            delay_between_days=delay_between_days,
            sort=sort,
            source_lang=source_language
        )
    
            




        
    
    
# used only for testing purposes
def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()      # json doesnt allow certain date formats so need to convert
    raise TypeError(...)



if __name__ == "__main__":
    collector = GDELTCollector()
    articles = collector.extract_multiple_days_using_ticker(
        ticker='NVDA', days_backwards=2, max_records_per_day=50, delay_between_days=6.0
    )

    df = convert_articles_to_dataframe(articles)
    if df is not None and not df.empty:
        df.to_csv('news_articles.csv')
        print(f"\nSaved {len(df)} articles to news_articles.csv")
    else:
        print("No articles to save")

