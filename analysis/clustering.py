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
    """runs clustering on daily feature matrix to find market sentiment regimes"""

    # these are the features we actually want to cluster on
    # dont want to include stuff like dates or article counts that would mess up the clustering
    DEFAULT_FEATURES = [
        'vader_mean',
        'vader_std',
        'vader_median',
        'positive_ratio',
        'negative_ratio',
        'daily_return',
        'realised_volatility_5d',
    ]

    def __init__(self,
                 min_cluster_size: int = 5,
                 min_samples: int = 3,
                 features: Optional[List[str]] = None):
        
        # min_cluster_size: smallest group HDBSCAN will consider a cluster 
         #     (smaller = more clusters, might be noisy tho)
        """
        min_cluster_size: smallest group HDBSCAN will consider a cluster 
                          (smaller = more clusters, might be noisy tho)
        min_samples: how conservative the clustering is 
                     (higher = more points become noise/-1)
        features: which columns to use, defaults to the ones above
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.features = features or self.DEFAULT_FEATURES
        self.scaler = StandardScaler()      # need to scale features or HDBSCAN freaks out
        self.clusterer = None               # gets set when we actually run fit()
        self.fitted = False

    def fit_predict(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        
        """
        main method - takes the feature matrix from feature_aggregate.py,
        scales it, runs HDBSCAN, and returns the df with cluster labels added

        returns the same dataframe but with extra columns:
            - cluster_label: which cluster each day belongs to (-1 = noise/outlier)
            - cluster_probability: how confident HDBSCAN is about the assignment
        """
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
        """
        builds a nice summary table of each cluster's characteristics
        useful for the dissertation writeup and for the visualisations
        """
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


# convenience function so we dont have to instantiate the class every time
def run_clustering(feature_matrix: pd.DataFrame,
                   min_cluster_size: int = 5,
                   min_samples: int = 3) -> Tuple[pd.DataFrame, SentimentClusterer]:
    """
    quick way to run clustering - returns the labelled dataframe and the clusterer object
    handy for the pipeline where we just want to call one function
    """
    clusterer = SentimentClusterer(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
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