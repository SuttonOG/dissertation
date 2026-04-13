
# Tests for the following
# test 1 - ticker_lookup query building
# test 2 - extracting articles - gdelt_collect
# test 3 - converting dataframes - article_to_dataframe
# test 4 - scoring sentiment - sentiment_vader
# test 5 - price data test - price_data
# test 6 - aggregating features - feature_aggregate
# test 7 - K-Means clustering - clustering_kmeans
# test 8 - FinBERT feature aggregation - feature_aggregate with finbert
# test 9 - sentiment-aware clustering - both methods x both scorers


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
    aggregate_daily_sentiment,
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
        dates = pd.bdate_range(end="2026-03-10", periods=20)
        close = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        

        mock_dl.return_value = pd.DataFrame({
            "Open": close - 0.5, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": [10_000_000] * len(dates),
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
        agg = aggregate_daily_sentiment(df)
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


# test 7 - K-Means clustering
class TestKMeansClustering:

    # verify K-Means produces cluster labels on a synthetic feature matrix
    def test_kmeans_assigns_labels(self):

        from analysis.clustering_kmeans import KMeansClusterer

        # build synthetic feature matrix with 20 data points, 2 clear clusters
        np.random.seed(42)
        df = pd.DataFrame({
            'vader_mean': np.concatenate([np.random.normal(0.3, 0.1, 10), np.random.normal(-0.3, 0.1, 10)]),
            'vader_std': np.random.uniform(0.05, 0.3, 20),
            'vader_median': np.concatenate([np.random.normal(0.25, 0.1, 10), np.random.normal(-0.25, 0.1, 10)]),
            'positive_ratio': np.concatenate([np.random.uniform(0.5, 0.8, 10), np.random.uniform(0.1, 0.4, 10)]),
            'negative_ratio': np.concatenate([np.random.uniform(0.1, 0.3, 10), np.random.uniform(0.4, 0.7, 10)]),
            'daily_return': np.concatenate([np.random.normal(0.01, 0.005, 10), np.random.normal(-0.01, 0.005, 10)]),
            'realised_volatility_5d': np.random.uniform(0.005, 0.02, 20),
        })

        clusterer = KMeansClusterer(n_clusters=2)
        result = clusterer.fit_predict(df)

        assert 'cluster_label' in result.columns
        assert 'cluster_probability' in result.columns
        assert result['cluster_label'].nunique() == 2           # should find exactly 2 clusters
        assert (result['cluster_label'] >= 0).all()             # no noise points in K-Means


    # verify auto k selection works and picks a reasonable k
    def test_kmeans_auto_k_selection(self):

        from analysis.clustering_kmeans import KMeansClusterer

        np.random.seed(42)
        df = pd.DataFrame({
            'vader_mean': np.concatenate([np.random.normal(0.3, 0.1, 15),
                                          np.random.normal(-0.3, 0.1, 15),
                                          np.random.normal(0.0, 0.05, 15)]),
            'vader_std': np.random.uniform(0.05, 0.3, 45),
            'daily_return': np.concatenate([np.random.normal(0.01, 0.005, 15),
                                            np.random.normal(-0.01, 0.005, 15),
                                            np.random.normal(0.0, 0.003, 15)]),
            'realised_volatility_5d': np.random.uniform(0.005, 0.02, 45),
        })

        clusterer = KMeansClusterer(n_clusters=None, max_k=6)
        result = clusterer.fit_predict(df)

        assert clusterer.best_k_ is not None
        assert 2 <= clusterer.best_k_ <= 6
        assert len(clusterer.silhouette_scores_) > 0
        assert len(clusterer.inertias_) > 0


    # verify get_cluster_profiles returns correct structure
    def test_kmeans_cluster_profiles(self):

        from analysis.clustering_kmeans import KMeansClusterer

        np.random.seed(42)
        df = pd.DataFrame({
            'vader_mean': np.random.normal(0.0, 0.2, 20),
            'vader_std': np.random.uniform(0.05, 0.3, 20),
            'daily_return': np.random.normal(0.0, 0.01, 20),
            'realised_volatility_5d': np.random.uniform(0.005, 0.02, 20),
        })

        clusterer = KMeansClusterer(n_clusters=2)
        result = clusterer.fit_predict(df)
        profiles = clusterer.get_cluster_profiles(result)

        assert not profiles.empty
        assert 'day_count' in profiles.columns
        assert len(profiles) == 2               # one row per cluster


    # verify evaluation metrics are computed
    def test_kmeans_evaluation_metrics(self):

        from analysis.clustering_kmeans import KMeansClusterer

        np.random.seed(42)
        df = pd.DataFrame({
            'vader_mean': np.random.normal(0.0, 0.2, 30),
            'vader_std': np.random.uniform(0.05, 0.3, 30),
            'daily_return': np.random.normal(0.0, 0.01, 30),
            'realised_volatility_5d': np.random.uniform(0.005, 0.02, 30),
        })

        clusterer = KMeansClusterer(n_clusters=3)
        clusterer.fit_predict(df)

        assert 'silhouette' in clusterer.evaluation_metrics_
        assert 'calinski_harabasz' in clusterer.evaluation_metrics_
        assert 'davies_bouldin' in clusterer.evaluation_metrics_
        assert 'inertia' in clusterer.evaluation_metrics_
        assert -1 <= clusterer.evaluation_metrics_['silhouette'] <= 1


# test 8 - FinBERT feature aggregation
class TestFinBERTAggregation:

    # verify aggregate_daily_sentiment works with finbert compound column
    def test_finbert_aggregation_produces_correct_columns(self):

        df = pd.DataFrame({
            "published_day": ["2026-03-10", "2026-03-10", "2026-03-09"],
            "finbert_compound": [0.6, 0.4, -0.3],
        })
        agg = aggregate_daily_sentiment(df, sentiment='finbert')

        assert len(agg) == 2
        assert "finbert_mean" in agg.columns
        assert "finbert_std" in agg.columns
        assert "finbert_median" in agg.columns
        assert "article_count" in agg.columns
        assert "positive_ratio" in agg.columns
        assert "negative_ratio" in agg.columns

        # vader columns should NOT be present
        assert "vader_mean" not in agg.columns


    # verify finbert aggregation computes correct values
    def test_finbert_aggregation_correct_values(self):

        df = pd.DataFrame({
            "published_day": ["2026-03-10", "2026-03-10"],
            "finbert_compound": [0.6, 0.4],
        })
        agg = aggregate_daily_sentiment(df, sentiment='finbert')

        assert len(agg) == 1
        assert abs(agg['finbert_mean'].iloc[0] - 0.5) < 0.001          # mean of 0.6 and 0.4


    # verify build_feature_matrix works with finbert sentiment
    def test_build_feature_matrix_finbert(self, sample_price_df):

        df = pd.DataFrame({
            "published_day": ["2026-03-09", "2026-03-10"],
            "finbert_compound": [0.4, -0.1],
        })
        matrix = build_feature_matrix(df, sample_price_df, sentiment='finbert')

        assert "finbert_mean" in matrix.columns
        assert "daily_return" in matrix.columns
        assert "vader_mean" not in matrix.columns


    # verify error raised when compound column missing
    def test_aggregation_raises_on_missing_column(self):

        df = pd.DataFrame({
            "published_day": ["2026-03-10"],
            "vader_compound": [0.5],           # only vader present
        })

        with pytest.raises(ValueError, match="finbert_compound"):
            aggregate_daily_sentiment(df, sentiment='finbert')


    # verify unknown sentiment scorer raises error
    def test_aggregation_raises_on_unknown_sentiment(self):

        df = pd.DataFrame({
            "published_day": ["2026-03-10"],
            "vader_compound": [0.5],
        })

        with pytest.raises(ValueError, match="Unknown sentiment"):
            aggregate_daily_sentiment(df, sentiment='unknown_scorer')


# test 9 - sentiment-aware clustering (both methods x both scorers)
class TestSentimentAwareClustering:

    # verify run_clustering with method='kmeans' and sentiment='vader'
    def test_run_clustering_kmeans_vader(self):

        from analysis.clustering import run_clustering

        np.random.seed(42)
        df = pd.DataFrame({
            'vader_mean': np.random.normal(0.0, 0.2, 20),
            'vader_std': np.random.uniform(0.05, 0.3, 20),
            'daily_return': np.random.normal(0.0, 0.01, 20),
            'realised_volatility_5d': np.random.uniform(0.005, 0.02, 20),
        })

        result, clusterer = run_clustering(df, method='kmeans', n_clusters=2, sentiment='vader')

        assert 'cluster_label' in result.columns
        assert result['cluster_label'].nunique() == 2
        assert 'vader_mean' in clusterer.features


    # verify run_clustering with method='kmeans' and sentiment='finbert'
    def test_run_clustering_kmeans_finbert(self):

        from analysis.clustering import run_clustering

        np.random.seed(42)
        df = pd.DataFrame({
            'finbert_mean': np.random.normal(0.0, 0.2, 20),
            'finbert_std': np.random.uniform(0.05, 0.3, 20),
            'finbert_median': np.random.normal(0.0, 0.15, 20),
            'positive_ratio': np.random.uniform(0.2, 0.7, 20),
            'negative_ratio': np.random.uniform(0.1, 0.5, 20),
            'daily_return': np.random.normal(0.0, 0.01, 20),
            'realised_volatility_5d': np.random.uniform(0.005, 0.02, 20),
        })

        result, clusterer = run_clustering(df, method='kmeans', n_clusters=2, sentiment='finbert')

        assert 'cluster_label' in result.columns
        assert result['cluster_label'].nunique() == 2
        assert 'finbert_mean' in clusterer.features
        assert 'vader_mean' not in clusterer.features


    # verify run_clustering with method='hdbscan' and sentiment='finbert'
    def test_run_clustering_hdbscan_finbert(self):

        from analysis.clustering import run_clustering

        np.random.seed(42)
        df = pd.DataFrame({
            'finbert_mean': np.concatenate([np.random.normal(0.4, 0.05, 15), np.random.normal(-0.3, 0.05, 15)]),
            'finbert_std': np.random.uniform(0.05, 0.2, 30),
            'finbert_median': np.concatenate([np.random.normal(0.35, 0.05, 15), np.random.normal(-0.25, 0.05, 15)]),
            'positive_ratio': np.concatenate([np.random.uniform(0.6, 0.9, 15), np.random.uniform(0.1, 0.3, 15)]),
            'negative_ratio': np.concatenate([np.random.uniform(0.05, 0.2, 15), np.random.uniform(0.5, 0.8, 15)]),
            'daily_return': np.concatenate([np.random.normal(0.01, 0.003, 15), np.random.normal(-0.01, 0.003, 15)]),
            'realised_volatility_5d': np.random.uniform(0.005, 0.015, 30),
        })

        result, clusterer = run_clustering(df, min_cluster_size=3, min_samples=2, method='hdbscan', sentiment='finbert')

        assert 'cluster_label' in result.columns
        assert 'finbert_mean' in clusterer.features
        assert 'vader_mean' not in clusterer.features


    # verify that finbert clusterer ignores vader columns even if present
    def test_finbert_clustering_ignores_vader_columns(self):

        from analysis.clustering_kmeans import KMeansClusterer

        np.random.seed(42)
        df = pd.DataFrame({
            # both vader and finbert columns present
            'vader_mean': np.random.normal(0.0, 0.2, 20),
            'finbert_mean': np.random.normal(0.1, 0.2, 20),
            'vader_std': np.random.uniform(0.05, 0.3, 20),
            'finbert_std': np.random.uniform(0.05, 0.3, 20),
            'finbert_median': np.random.normal(0.05, 0.15, 20),
            'positive_ratio': np.random.uniform(0.2, 0.7, 20),
            'negative_ratio': np.random.uniform(0.1, 0.5, 20),
            'daily_return': np.random.normal(0.0, 0.01, 20),
            'realised_volatility_5d': np.random.uniform(0.005, 0.02, 20),
        })

        clusterer = KMeansClusterer(n_clusters=2, sentiment='finbert')
        result = clusterer.fit_predict(df)

        # should use finbert features, not vader
        assert 'finbert_mean' in clusterer.features
        assert 'vader_mean' not in clusterer.features
        assert result['cluster_label'].nunique() == 2


    # verify backward compatibility - no sentiment arg defaults to vader
    def test_default_sentiment_is_vader(self):

        from analysis.clustering import run_clustering

        np.random.seed(42)
        df = pd.DataFrame({
            'vader_mean': np.random.normal(0.0, 0.2, 20),
            'vader_std': np.random.uniform(0.05, 0.3, 20),
            'daily_return': np.random.normal(0.0, 0.01, 20),
            'realised_volatility_5d': np.random.uniform(0.005, 0.02, 20),
        })

        # no sentiment arg passed - should default to vader
        result, clusterer = run_clustering(df, method='kmeans', n_clusters=2)

        assert 'vader_mean' in clusterer.features
        assert clusterer.sentiment == 'vader'


    # verify invalid method raises ValueError
    def test_invalid_method_raises_error(self):

        from analysis.clustering import run_clustering

        df = pd.DataFrame({'vader_mean': [0.1], 'daily_return': [0.01]})

        with pytest.raises(ValueError, match="Unknown clustering method"):
            run_clustering(df, method='invalid_method')
