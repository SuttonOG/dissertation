# clustering_kmeans.py - K-Means clustering on the feature matrix
# baseline comparison model for the dissertation
# unlike HDBSCAN, K-Means requires specifying k upfront
# uses elbow method + silhouette scores to find optimal k
# same interface as SentimentClusterer for easy swapping in the pipeline

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Optional, List, Tuple


class KMeansClusterer:

    # Kmeans clustering for daily feature matrix -> distinguish market sentiment regimes

    # baseline comparison model vs other models
    # same interface as sentimentClusterer fit_predict() and get_cluster_profiles()

    # differences to hdbscan - req n_cluster specification, no noise point
    # includes auto k-selection via silhouette score if k not specified.


    # feature sets per sentiment scorer (same as HDBSCAN for fair comparison)
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
                 n_clusters: Optional[int] = None,
                 max_k: int = 8,
                 random_state: int = 42,
                 features: Optional[List[str]] = None,
                 sentiment: str = 'vader'):
        """
        n_clusters: number of clusters (k). If None, auto-selects best k
                    using silhouette score over range 2..max_k
        max_k: upper bound for k search when n_clusters is None
        random_state: seed for reproducibility
        features: which columns to cluster on. If None, auto-selects based on sentiment param
        sentiment: which scorer was used ('vader' or 'finbert'), determines default features
        """
        self.n_clusters = n_clusters
        self.max_k = max_k
        self.random_state = random_state
        self.sentiment = sentiment
        self.features = features or self.FEATURE_SETS.get(sentiment, self.DEFAULT_FEATURES)
        self.scaler = StandardScaler()
        self.model = None
        self.fitted = False

        # store evaluation metrics after fitting
        self.inertias_ = {}             # SSE for each k tested (for elbow plot)
        self.silhouette_scores_ = {}    # silhouette score for each k tested
        self.best_k_ = None            # the k that was actually used
        self.evaluation_metrics_ = {}   # final model metrics


    def _find_optimal_k(self, X_scaled: np.ndarray) -> int:
        """Find the best k using silhouette score.
        
        Tests k from 2 to max_k, returns the k with highest silhouette score.
        Also stores inertia values for elbow plot visualisation.
        """
        print(f"  Auto-selecting k (testing k=2..{self.max_k})...")

        best_k = 2
        best_score = -1

        for k in range(2, self.max_k + 1):
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10,
                max_iter=300,
            )
            labels = kmeans.fit_predict(X_scaled)

            # store inertia for elbow method
            self.inertias_[k] = kmeans.inertia_

            # silhouette score (-1 to 1, higher = better separated clusters)
            score = silhouette_score(X_scaled, labels)
            self.silhouette_scores_[k] = score

            print(f"    k={k}: silhouette={score:.4f}, inertia={kmeans.inertia_:.2f}")

            if score > best_score:
                best_score = score
                best_k = k

        print(f"  Best k={best_k} (silhouette={best_score:.4f})")
        return best_k


    def fit_predict(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:

        # main method - scales features -> run kmeans -> return df with cluster labels
        # same interface as sentimentclusterer.fit_predict()
        # return same df withe xtra cols

        # cluster_label = which day cluster belongs to
        # cluster_probability = distance to cluster centre (invert + normalised) so higher = closer to centre.

        df = feature_matrix.copy()

        # figure out which features we can actually use
        available_features = [f for f in self.features if f in df.columns]

        if len(available_features) < 2:
            print(f"ERROR: need at least 2 features to cluster, only found {available_features}")
            df['cluster_label'] = -1
            df['cluster_probability'] = 0.0
            return df

        if len(available_features) < len(self.features):
            missing = set(self.features) - set(available_features)
            print(f"  heads up: missing features {missing}, using what we have")

        print(f"\nK-Means clustering with {len(available_features)} features: {available_features}")
        print(f"  data points (trading days): {len(df)}")

        # grab feature columns and drop NaN rows
        X = df[available_features].copy()
        valid_mask = X.notna().all(axis=1)
        X_clean = X[valid_mask]

        if len(X_clean) < 3:
            print(f"  not enough valid data points ({len(X_clean)}) for K-Means clustering")
            df['cluster_label'] = -1
            df['cluster_probability'] = 0.0
            return df

        dropped = len(X) - len(X_clean)
        if dropped > 0:
            print(f"  dropped {dropped} rows with NaN values")

        # scale features (same as HDBSCAN)
        X_scaled = self.scaler.fit_transform(X_clean)

        # determine k
        if self.n_clusters is not None:
            k = self.n_clusters
            # cap k at number of data points
            if k >= len(X_clean):
                k = max(2, len(X_clean) - 1)
                print(f"  adjusted k to {k} (not enough data points for k={self.n_clusters})")
        else:
            # auto-select k, but cap max_k at data points - 1
            actual_max_k = min(self.max_k, len(X_clean) - 1)
            if actual_max_k < 2:
                actual_max_k = 2
            self.max_k = actual_max_k
            k = self._find_optimal_k(X_scaled)

        self.best_k_ = k

        # fit final model
        self.model = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
        )
        labels = self.model.fit_predict(X_scaled)

        # compute cluster probability as inverse normalised distance to centroid
        # so it matches HDBSCAN output format (higher = more confident assignment)
        distances = self.model.transform(X_scaled)            # distance to each centroid
        assigned_distances = distances[np.arange(len(labels)), labels]   # distance to assigned centroid
        max_dist = assigned_distances.max() if assigned_distances.max() > 0 else 1.0
        probabilities = 1.0 - (assigned_distances / max_dist)

        # compute evaluation metrics for final model
        self.evaluation_metrics_ = {
            'silhouette': silhouette_score(X_scaled, labels),
            'calinski_harabasz': calinski_harabasz_score(X_scaled, labels),
            'davies_bouldin': davies_bouldin_score(X_scaled, labels),
            'inertia': self.model.inertia_,
        }

        # add results back to dataframe
        # K-Means assigns every point, but NaN rows still get -1 for consistency
        df['cluster_label'] = -1
        df['cluster_probability'] = 0.0
        df.loc[valid_mask, 'cluster_label'] = labels
        df.loc[valid_mask, 'cluster_probability'] = probabilities

        self.fitted = True

        self._print_summary(df)

        return df


    def _print_summary(self, df: pd.DataFrame):
        """Print clustering results summary - same format as HDBSCAN for comparison."""

        n_clusters = df[df['cluster_label'] >= 0]['cluster_label'].nunique()
        noise_count = (df['cluster_label'] == -1).sum()
        total = len(df)

        print(f"\n{'=' * 50}")
        print(f"K-MEANS CLUSTERING RESULTS")
        print(f"{'=' * 50}")
        print(f"  k (clusters): {n_clusters}")
        print(f"  unassigned:   {noise_count}/{total} ({100*noise_count/total:.0f}%)")

        # print evaluation metrics
        if self.evaluation_metrics_:
            print(f"\n  Evaluation metrics:")
            print(f"    silhouette score:       {self.evaluation_metrics_['silhouette']:.4f}")
            print(f"    calinski-harabasz:      {self.evaluation_metrics_['calinski_harabasz']:.2f}")
            print(f"    davies-bouldin:         {self.evaluation_metrics_['davies_bouldin']:.4f}")
            print(f"    inertia (SSE):          {self.evaluation_metrics_['inertia']:.2f}")

        # show each cluster's characteristics
        for label in sorted(df['cluster_label'].unique()):
            if label == -1:
                continue        # skip unassigned (only NaN rows)

            cluster_df = df[df['cluster_label'] == label]
            print(f"\n  Cluster {label} ({len(cluster_df)} days):")

            if 'vader_mean' in cluster_df.columns:
                print(f"    avg sentiment:  {cluster_df['vader_mean'].mean():.4f}")
            if 'daily_return' in cluster_df.columns:
                print(f"    avg return:     {cluster_df['daily_return'].mean():.4f}")
            if 'realised_volatility_5d' in cluster_df.columns:
                print(f"    avg volatility: {cluster_df['realised_volatility_5d'].mean():.4f}")
            if 'positive_ratio' in cluster_df.columns:
                print(f"    pos/neg ratio:  {cluster_df['positive_ratio'].mean():.2f} / {cluster_df['negative_ratio'].mean():.2f}")


    def get_cluster_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build summary table of each cluster's characteristics.
        
        Same interface as SentimentClusterer.get_cluster_profiles().
        """
        if 'cluster_label' not in df.columns:
            print("no cluster labels found - run fit_predict first")
            return pd.DataFrame()

        clustered = df[df['cluster_label'] >= 0].copy()

        if clustered.empty:
            print("no clusters found")
            return pd.DataFrame()

        feature_cols = [f for f in self.features if f in df.columns]

        # group by cluster and get mean + std for each feature
        profiles = clustered.groupby('cluster_label')[feature_cols].agg(['mean', 'std'])

        counts = clustered.groupby('cluster_label').size().rename('day_count')

        # flatten multi-level columns
        profiles.columns = ['_'.join(col) for col in profiles.columns]
        profiles = profiles.join(counts)

        print(f"\nCluster profiles built for {len(profiles)} clusters")
        return profiles


# convenience function matching run_clustering interface
def run_kmeans_clustering(feature_matrix: pd.DataFrame,
                          n_clusters: Optional[int] = None,
                          max_k: int = 8,
                          random_state: int = 42) -> Tuple[pd.DataFrame, KMeansClusterer]:
    
    # quick way to run k-means
    # return labeled df + cluster object
    # if n_clusters = None, auto select best k via silhouette score.

    clusterer = KMeansClusterer(
        n_clusters=n_clusters,
        max_k=max_k,
        random_state=random_state,
    )
    result = clusterer.fit_predict(feature_matrix)
    return result, clusterer


# testing block
if __name__ == "__main__":
    import os

    test_path = "data/feature_matrix.csv"

    if os.path.exists(test_path):
        print(f"Loading feature matrix from {test_path}")
        df = pd.read_csv(test_path)

        # test 1: auto k selection
        print("\n" + "=" * 60)
        print("TEST 1: Auto k selection")
        print("=" * 60)
        result, clusterer = run_kmeans_clustering(df, max_k=6)

        output_path = "data/clustered_kmeans.csv"
        result.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")

        profiles = clusterer.get_cluster_profiles(result)
        if not profiles.empty:
            print(f"\n{profiles.to_string()}")

        # test 2: fixed k=3
        print("\n" + "=" * 60)
        print("TEST 2: Fixed k=3")
        print("=" * 60)
        result2, clusterer2 = run_kmeans_clustering(df, n_clusters=3)
        profiles2 = clusterer2.get_cluster_profiles(result2)
        if not profiles2.empty:
            print(f"\n{profiles2.to_string()}")

    else:
        print(f"no feature matrix found at {test_path}")
        print("run the pipeline first: python pipeline.py --ticker NVDA --days 30")
