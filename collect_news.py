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




#Define Article dataclass

@dataclass
class NewsArticle:
    #Store relevant news article information
    title: str
    summary : str
    source: str
    content : Optional[str] = None
    # relevance_score   -> to be added later when implemented
    # matches ->       to add later



