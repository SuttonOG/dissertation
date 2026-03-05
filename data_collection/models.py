#Define Article dataclass
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class NewsArticle:
    title: str
    source: str
    url: str
    published_date: datetime
    summary: str = ''                   # not all articles extract a summary
    content: Optional[str] = None
