
#This file will be used for the extraction of news articles from RSS feeeds and from relevant news articles


import feedparser                           # extracting RSS feeds articles
import pandas as pd                         # pandas for converting article objects to dataframe
import requests                             # used for extracting News API articles
from datetime import datetime, timedelta    # used for scraping specific periods of articles  
import time
import json
import re
import numpy as np
from dataclasses import dataclass, field    # dataclass used for creating immutable article objects
from typing import List, Dict, Optional, Set, Tuple             
import yfinance as yf                           # extracting stock data

from dataclasses import asdict


#Define Article dataclass

@dataclass
class NewsArticle:
    title: str
    summary: str
    source: str
    url: str
    published_date: datetime
    content: Optional[str] = None



class RSSFeedCollector:


    # used to extract from the chosen RSS feeds  - yahoo finance, ft_markets, ft_companies


    RSS_Feeds = {   
        'ft_markets' : 'https://www.ft.com/rss/markets',
        'yahoo_finance' : 'https://finance.yahoo.com/news/rssindex',
        'ft_companies' : 'https://www.ft.com/rss/companies',
    }


    def __init__(self, feeds : Optional[Dict[str,str]] = None):
        

        # set up custom feeds or use the default feeds otherwise


        self.feeds = feeds or self.RSS_Feeds
    

    # Invoke to extract articles from one RSS feed
    def collect_from_feed(self, source_name: str, feed_url : str) -> List[NewsArticle]:  
        

        #Create news article object from each entry from the feed, extend list of articles
        news_articles = []

        try:
            # extract dict of articles from link
            feed = feedparser.parse(feed_url, agent = 'Mozilla/5.0  (compatible; NewsCollector/1.0)')

            # for each article, get date published -> extract information for article object, and append article to news_articles
            for entry in feed.entries:
                try:
                    # extract the data it was published
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_date = datetime(*entry.published_parsed[:6])              # take first 5 values due to format of date
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published_date = datetime(*entry.updated_parsed[:6])
                    else:
                        published_date = datetime.now()


                    #Extract summary / desc from each article
                    summary = entry.get('summary', entry.get('description',''))

                    # clean 
                    summary = re.sub('<[^>]+>', '', summary)                     # remove any html characters from summary

                    summary = summary.strip()[:500]                             # RESTRICT length of summary


                    article = NewsArticle(
                        title = entry.get('title', ''),
                        summary = summary,
                        url = entry.get('link', ''),
                        published_date = published_date,
                        source = source_name,
                        content = entry.get('content', [{}])[0].get('value') if hasattr(entry, 'content') else None
                    )
                    
                    news_articles.append(article)                       # add article object to list of article object after extracting

                except Exception as e:
                    print(f"Error with parsing from {entry.title}, error {e}")
                    continue # continue on to the next article
            
        # After loop, all articles extracted
            print(f"Successful extraction of {len(news_articles)} from {source_name}")

        except Exception as e:
            print(f"Error extracting from {source_name}. Error code {e}")

        return news_articles
    
    # collect from ALL RSS Feeds

    def collect_from_all_feeds(self, days_backwards):

        # each collect_from_feed returns a list of NewsArticle Objects from a feed, we want to extend all_articles with these lists

        all_articles = []
        cutoff_date = datetime.now() - timedelta(days=days_backwards)           # stop collecting historical articles after X days
        

        for source_name, feed_url in self.feeds.items():            # returns format example   source_name = ft_market, feed_url = url 

            #run per url
            specific_feed_articles = self.collect_from_feed(source_name, feed_url)

            #filter the articles by ensuring their dates >= cutoff_date
            filtered_articles = [a for a in specific_feed_articles if a.published_date >= cutoff_date]

            all_articles.extend(filtered_articles)
            time.sleep(1) 

        #Return list of news article objects from ALL RSS_FEEDS      
        return all_articles 
            






if __name__ == "__main__":
    # used for testing purposes for now

    test_feeds = {'ft_markets' : 'https://www.ft.com/rss/markets'}

    collector = RSSFeedCollector()  

    all_articles = collector.collect_from_all_feeds(days_backwards=1)       

    with open('test2.json','w') as f:
        json.dump([asdict(a) for a in all_articles], f, default = str, indent = 2 )

        # test run - should return a list of newsarticle objects to dump into json

        
            # articles = collector.collect_from_feed(source_name = 'ft_markets',feed_url ='https://www.ft.com/rss/markets')
            # with open('test.json', 'w') as f:
            #     json.dump([asdict(a) for a in articles], f, default= str, indent = 4)
        
    # Test run of collect_from_all_feeds

