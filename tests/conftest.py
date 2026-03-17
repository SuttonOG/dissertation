

import pytest
import pandas as pd
from datetime import datetime, timedelta
from data_collection.models import NewsArticle


@pytest.fixture
def sample_articles():


    # use a set of test articles, contains one duplicate for testing deduping
    base_date = datetime(2026, 3, 10)
    return [
        #articel 1
        NewsArticle(
            title="NVIDIA stock surges after record earnings",
            source="gdelt_reuters.com",
            url="https://reuters.com/nvidia-surge",
            published_date=base_date,
            content="NVIDIA shares jumped 8% after blowout earnings.",
        ),
        # article 2 -
        NewsArticle(
            title="Markets crash as trade war fears intensify",
            source="gdelt_bbc.com",
            url="https://bbc.com/markets-crash",
            published_date=base_date - timedelta(days=1),
            content=None,
        ),
        # article 3
        NewsArticle(
            title="Tech giant reports strong cloud growth",
            source="rss_yahoo",
            url="https://yahoo.com/tech-cloud",
            published_date=base_date,
            content="Cloud revenue grew 40% year over year.",
        ),

        # duplicate of first article contains same title + same day
        NewsArticle(
            title="NVIDIA stock surges after record earnings",
            source="gdelt_cnbc.com",
            url="https://cnbc.com/nvidia-dup",
            published_date=base_date,
            content="Duplicate content.",
        ),
    ]


@pytest.fixture
def sample_price_df():
    
    # mock price data matching shape produced from price_data
    dates = pd.bdate_range(end="2026-03-10", periods=10)
    df = pd.DataFrame({
        "Close": [100 + i for i in range(10)],
        "daily_return": [0.005 * ((-1) ** i) for i in range(10)],
        "log_return": [0.005 * ((-1) ** i) for i in range(10)],
        "realised_volatility_5d": [0.012] * 10,
        "realised_volatility_20d": [0.015] * 10,
    }, index=dates)
    df["date"] = df.index.date
    return df