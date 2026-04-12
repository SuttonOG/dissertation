# clustering.py - runs HDBSCAN clustering on the feature matrix
# basically groups trading days into "regimes" based on sentiment + price behaviour
# e.g. a cluster might be "high negative sentiment + high volatility" days
# uses sklearn for scaling and HDBSCAN for the actual clustering

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict, Tuple
import hdbscan


class SentimentClusterer:
  
    # runs clustering on daily feature matrix produced to find market sentiment regimes

    # feature sets per sentiment scorer
    # the sentiment columns match the output of aggregate_daily_sentiment
    FEATURE_SETS = {
        'vader': [
            'vader_mean',
            'vader_std',
            'vader_median',
            'positive_ratio',
            'negative_ratio',
            'daily_return',
            'realised_volatility_5d',
        ],
        'finbert': [
            'finbert_mean',
            'finbert_std',
            'finbert_median',
            'positive_ratio',
            'negative_ratio',
            'daily_return',
            'realised_volatility_5d',
        ],
    }

    # keep DEFAULT_FEATURES for backward compatibility (defaults to vader)
    DEFAULT_FEATURES = FEATURE_SETS['vader']

    def __init__(self,
                 min_cluster_size: int = 5,
                 min_samples: int = 3,
                 features: Optional[List[str]] = None,
                 sentiment: str = 'vader'):
        

        self.min_cluster_size = min_cluster_size                            # smallest group HDBSCAN will consider cluster. small = more clust, but might be noisy             
        self.min_samples = min_samples                                      # how conservative clustering is, high = more points become noise
        self.sentiment = sentiment
        self.features = features or self.FEATURE_SETS.get(sentiment, self.DEFAULT_FEATURES)
        self.scaler = StandardScaler()      # need to scale features or HDBSCAN freaks out
        self.clusterer = None               # gets set when we actually run fit()
        self.fitted = False

    def fit_predict(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:

        # main method - takes feature matrix from feature_aggregate.py -> scales -> runs hdbscan -> returns df + cluster labels
        
        # return same df but with extra cols (cluster_label - the cluster a day belongs to (-1 is outlier) and cluster_probability (how confident HDBSCAN is))

        df = feature_matrix.copy()

        # figure out which features we can actually use (some might be missing)
        available_features = [f for f in self.features if f in df.columns]

        if len(available_features) < 2:
            print(f"ERROR: need at least 2 features to cluster, only found {available_features}")
            print(f"  available columns: {list(df.columns)}")
            df['cluster_label'] = -1
            df['cluster_probability'] = 0.0
            return df

        if len(available_features) < len(self.features):
            missing = set(self.features) - set(available_features)
            print(f"  heads up: missing features {missing}, using what we have")

        print(f"\nClustering with {len(available_features)} features: {available_features}")
        print(f"  data points (trading days): {len(df)}")

        # grab just the feature columns and drop any rows with NaN
        # (happens sometimes at the start of the data where rolling calcs havent kicked in)
        X = df[available_features].copy()
        valid_mask = X.notna().all(axis=1)
        X_clean = X[valid_mask]

        # debugging - ensure dataframe size is >= cluster size, or unable to assign any clusters
        if len(X_clean) < self.min_cluster_size:
            print(f"  not enough valid data points ({len(X_clean)}) for clustering")
            print(f"  need at least {self.min_cluster_size} - try collecting more days of data")
            df['cluster_label'] = -1
            df['cluster_probability'] = 0.0
            return df

        dropped = len(X) - len(X_clean)
        if dropped > 0:
            print(f"  dropped {dropped} rows with NaN values")

        # IMPORTANT: scale the features first
        # without this, a feature like daily_return (tiny numbers like 0.02)
        # gets completely dominated by article_count (big numbers like 50)
        X_scaled = self.scaler.fit_transform(X_clean)

        # run HDBSCAN - the main event
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
        )
        labels = self.clusterer.fit_predict(X_scaled)
        probabilities = self.clusterer.probabilities_

        # add results back to the dataframe
        # days that got dropped (NaN) get labelled as noise (-1)
        df['cluster_label'] = -1
        df['cluster_probability'] = 0.0
        df.loc[valid_mask, 'cluster_label'] = labels
        df.loc[valid_mask, 'cluster_probability'] = probabilities

        self.fitted = True

        # print a summary so we can see whats going on
        self._print_summary(df)

        return df

    def _print_summary(self, df: pd.DataFrame):
        """prints out what the clustering found - useful for debugging and demo"""

        n_clusters = df[df['cluster_label'] >= 0]['cluster_label'].nunique()
        noise_count = (df['cluster_label'] == -1).sum()
        total = len(df)

        print(f"\n{'=' * 50}")
        print(f"CLUSTERING RESULTS")
        print(f"{'=' * 50}")
        print(f"  clusters found: {n_clusters}")
        print(f"  noise points:   {noise_count}/{total} ({100*noise_count/total:.0f}%)")

        # show whats in each cluster
        for label in sorted(df['cluster_label'].unique()):
            cluster_df = df[df['cluster_label'] == label]
            name = f"Cluster {label}" if label >= 0 else "Noise"

            print(f"\n  {name} ({len(cluster_df)} days):")

            # show the key stats for this cluster so we can interpret it
            if 'vader_mean' in cluster_df.columns:
                print(f"    avg sentiment:  {cluster_df['vader_mean'].mean():.4f}")
            if 'daily_return' in cluster_df.columns:
                print(f"    avg return:     {cluster_df['daily_return'].mean():.4f}")
            if 'realised_volatility_5d' in cluster_df.columns:
                print(f"    avg volatility: {cluster_df['realised_volatility_5d'].mean():.4f}")
            if 'positive_ratio' in cluster_df.columns:
                print(f"    pos/neg ratio:  {cluster_df['positive_ratio'].mean():.2f} / {cluster_df['negative_ratio'].mean():.2f}")

    def get_cluster_profiles(self, df: pd.DataFrame) -> pd.DataFrame:

        # build summary table of each clusters characteristics for write up

        if 'cluster_label' not in df.columns:
            print("no cluster labels found - run fit_predict first")
            return pd.DataFrame()

        # only look at actual clusters, not noise
        clustered = df[df['cluster_label'] >= 0].copy()

        if clustered.empty:
            print("no clusters found (everything is noise) - try adjusting min_cluster_size")
            return pd.DataFrame()

        # get the feature columns that exist
        feature_cols = [f for f in self.features if f in df.columns]

        # group by cluster and get mean + std for each feature
        profiles = clustered.groupby('cluster_label')[feature_cols].agg(['mean', 'std'])

        # also add the count
        counts = clustered.groupby('cluster_label').size().rename('day_count')
        
        # flatten the multi-level columns cos they're annoying to work with
        profiles.columns = ['_'.join(col) for col in profiles.columns]
        profiles = profiles.join(counts)

        print(f"\nCluster profiles built for {len(profiles)} clusters")
        return profiles


# convenience function so dont have to instantiate the class every time
def run_clustering(feature_matrix: pd.DataFrame,
                   min_cluster_size: int = 5,
                   min_samples: int = 3,
                   method: str = 'hdbscan',
                   n_clusters: Optional[int] = None,
                   max_k: int = 8,
                   random_state: int = 42,
                   sentiment: str = 'vader') -> Tuple[pd.DataFrame, object]:
    
    
    # returns labelled df + clusterer object
    # handy for pipeline for just calling one func
    # pass cluster method 'hdbscan' / 'kmeans'
    # sentiment: 'vader' or 'finbert' - determines which feature columns to use
    # max_k for kmeans upper bound, 
    # n_clusters for k-means no. clusters


    # k means method
    if method == 'kmeans':
        from analysis.clustering_kmeans import KMeansClusterer
        clusterer = KMeansClusterer(
            n_clusters=n_clusters,
            max_k=max_k,
            random_state=random_state,
            sentiment=sentiment,
        )
    # hdbscan method
    elif method == 'hdbscan':
        clusterer = SentimentClusterer(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            sentiment=sentiment,
        )
    else:
        raise ValueError(f"Unknown clustering method: '{method}'. Use 'hdbscan' or 'kmeans'.")

    result = clusterer.fit_predict(feature_matrix)
    return result, clusterer


# testing block
if __name__ == "__main__":
    import os

    # try to load a feature matrix if one exists from a previous pipeline run
    test_path = "data/feature_matrix.csv"

    if os.path.exists(test_path):
        print(f"Loading feature matrix from {test_path}")
        df = pd.read_csv(test_path)

        result, clusterer = run_clustering(df, min_cluster_size=3, min_samples=2)

        # save the clustered result
        output_path = "data/clustered_features.csv"
        result.to_csv(output_path, index=False)
        print(f"\nSaved clustered data to {output_path}")

        # print profiles
        profiles = clusterer.get_cluster_profiles(result)
        if not profiles.empty:
            print(f"\n{profiles.to_string()}")
    else:
        print(f"no feature matrix found at {test_path}")
        print("run the pipeline first: python pipeline.py --ticker NVDA --days 30")