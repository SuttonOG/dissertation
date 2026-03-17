
# Tests for the following
# test 1 - ticker_lookup query building
# test 2 - extracting articles - gdelt_collect
# test 3 - converting dataframes - article_to_dataframe
# test 4 - scoring sentiment - sentiment_vader
# test 5 - price data test - price_data
# test 6 - aggregating features - feature_aggregate


import pytest
import json
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from data_collection.models import NewsArticle
from data_collection.ticker_lookup import (
    remove_company_endings, collect_from_yf, build_query_for_gdelt,
)
from data_collection.gdelt_collect import GDELTCollector
from processing.article_to_dataframe import convert_articles_to_dataframe
from processing.sentiment_vader import VaderScorer
from data_collection.price_data import get_price_extracted_data
from processing.feature_aggregate import (
    aggregate_aggregated_daily_sentiment_df_sentiment,
    build_feature_matrix,
)

# test 1 - building query

class TestQueryBuilding:
    
        # verify ticker_lookup builds the correct query
    def test_company_ending_removal(self):
        assert remove_company_endings("Apple Inc.") == "Apple"
        assert remove_company_endings("Microsoft Corporation") == "Microsoft"
        assert remove_company_endings("NVIDIA") == "NVIDIA"

    # test of full route, yfinance to querypack to gdelt query
    @patch("data_collection.ticker_lookup.yf.Ticker")
    def test_collect_from_yf_builds_gdelt_query(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {
            "longName": "NVIDIA Corporation",
            "shortName": "NVIDIA Corp",
            "quoteType": "EQUITY",
            "sector": "Technology",
            "industry": "Semiconductors",
            "country": "United States",
            "exchange": "NMS",
            "longBusinessSummary": "NVIDIA designs GPUs.",
        }
        pack = collect_from_yf("NVDA")
        query = build_query_for_gdelt(asdict(pack))

        assert "NVDA" in query
        assert "NVIDIA" in query
        assert query.startswith("(") and query.endswith(")")



# Test 2 - extract_articles for GDELT
class TestArticleExtraction:
    
    # Verifies that GDELT collect correctly parses the API response into NewsArticles
    @patch("data_collection.gdelt_collect.requests.get")
    def test_extracts_articles_from_valid_response(self, mock_get):
        # mock via json response -> should make newsarticle objects

        mock_response = {
            "articles": [
                {"title": "NVIDIA beats earnings", "url": "https://example.com/nvidia",
                 "seendate": "20260310120000", "domain": "example.com"},
                {"title": "AI chip demand soars", "url": "https://example.com/ai",
                 "seendate": "20260309080000", "domain": "example.com"},
            ]
        }
       
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = json.dumps(mock_response)
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        collector = GDELTCollector()
        articles = collector.extract_articles(query="NVIDIA", timespan="7d")

        # verify correct output, 2 articles appended with correct titles
        assert len(articles) == 2
        assert articles[0].title == "NVIDIA beats earnings"
        assert isinstance(articles[0], NewsArticle)             # ensure newsarticle object is returned

    # test for when an empty GDELT response given, shouldnt crash, just returns []
    @patch("data_collection.gdelt_collect.requests.get")
    def test_empty_response_returns_empty_list(self, mock_get):
        
        # empty response mock
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""                 
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        collector = GDELTCollector()
        assert collector.extract_articles(query="NOTHING") == []



# test 3 - converting dataframes and deduping
class TestDataFrameConversion:
    
    # verifies that articles are converted and deduplicated correctly
    def test_converts_articles_to_dataframe(self, sample_articles):
        
        # expect a Dataframe with expected columns title, published_day and origin
        
        df = convert_articles_to_dataframe(sample_articles)
        
        assert isinstance(df, pd.DataFrame)
        assert "title" in df.columns                # verify title col
        assert "published_day" in df.columns        # verify published day present
        assert "origin" in df.columns               # verify origin 

    def test_removes_duplicate_articles(self, sample_articles):
        # sample article has 4 articles, one duplication, so assert only 3 are in final
        df = convert_articles_to_dataframe(sample_articles)
        assert len(df) == 3



# test 4 - sentiment scoring
class TestSentimentScoring:
    
    # verify that VADER produces scares in the expected ranges -1 0 +1
    def test_positive_vs_negative_text(self):
        
        # verify a test positive text score > negative tezt
        scorer = VaderScorer()
        
        positive_text = scorer.score_text("Stock surges after incredible earnings beat")          # positive >
        negative_text = scorer.score_text("Markets crash amid fears of severe recession")         # negative <
        assert positive_text["compound"] > negative_text["compound"]                              # assert pos > neg

    def test_score_dataframe_adds_columns(self):
        # ensure scoring df adds cols vader_compound + vader_source
        
        
        scorer = VaderScorer()
        df = pd.DataFrame({
            "title": ["Great results", "Terrible losses"],
            "content": [None, None],
        })
        scored_df = scorer.score_dataframe(df)
        
        
        assert "vader_compound" in scored_df.columns                # assert cols present
        assert "vader_source" in scored_df.columns
        assert len(scored_df) == 2                                  # assert col len == 2


# test 5 - price_data extraction 
class TestPriceData:
   
    # assert price extract returns expected cols
    @patch("data_collection.price_data.yf.download")
    def test_returns_dataframe_with_returns(self, mock_dl):
    
        
        # should return a daily_return and a log_return from ohlcv data
        dates = pd.bdate_range(end=datetime.now(), periods=20)
        close = 100 + np.cumsum(np.random.randn(20) * 0.5)
        
        

        mock_dl.return_value = pd.DataFrame({
            "Open": close - 0.5, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": [10_000_000] * 20,
        }, index=dates)


        df = get_price_extracted_data(ticker="NVDA", days_back=15)
        assert df is not None
        assert "daily_return" in df.columns
        assert "log_return" in df.columns

    # test for when if yfanince returns nothing, so should function
    @patch("data_collection.price_data.yf.download")
    def test_empty_download_returns_none(self, mock_dl):
        
        mock_dl.return_value = pd.DataFrame()
        assert get_price_extracted_data(ticker="FAKE") is None


# test 6 - featurer aggregating
class TestFeatureAggregation:
        
    # verify that daily sentiment gets aggregated and merged with price data


    def test_aggregates_by_day(self):
        

        # test that 2 articles on same day produce 1 aggregated row
        df = pd.DataFrame({
            "published_day": ["2026-03-10", "2026-03-10", "2026-03-09"],        # 2 articles same day, one article another
            "vader_compound": [0.5, 0.3, -0.2],
        })
        agg = aggregate_aggregated_daily_sentiment_df_sentiment(df)
        assert len(agg) == 2                                                    # assert len == 2 , ensures the 2 same day have aggregated

        # assert cols present
        assert "vader_mean" in agg.columns                                      
        assert "article_count" in agg.columns


    def test_build_feature_matrix_with_prices(self, sample_price_df):
        

        # verify that the feature matrix contains sentiment & price cols
        df = pd.DataFrame({
            "published_day": ["2026-03-09", "2026-03-10"],
            "vader_compound": [0.4, -0.1],
        })
        matrix = build_feature_matrix(df, sample_price_df)

        # assert cols present
        assert "vader_mean" in matrix.columns
        assert "daily_return" in matrix.columns
