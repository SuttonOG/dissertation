# Gaussian Mixture Model clustering on the feature matrix
# its a soft clustering model that gives probability of belonging to each cluster
# uses BIC to auto-select the number of components if not specified
# uses same interface as SentimentClusterer + KMeansClusterer

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Optional, List, Tuple




class GMMClusterer:

    # GMM clustering on the daily feature matrix
    # key advantage over K-Means: soft (probabilistic) cluster assignments
    # each day gets a probability of belonging to each cluster instead of a hard label
    # also allows elliptical cluster shapes via full covariance matrices (something k-means cant do as predicts only blobs)

    # same feature sets as other clusterers for fair comparison
    FEATURE_SETS = {
        'vader': [
            'vader_mean', 'vader_std', 'vader_median',
            'positive_ratio', 'negative_ratio',
            'daily_return', 'realised_volatility_5d',
        ],
        'finbert': [
            'finbert_mean', 'finbert_std', 'finbert_median',
            'positive_ratio', 'negative_ratio',
            'daily_return', 'realised_volatility_5d',
        ],
    }



    DEFAULT_FEATURES = FEATURE_SETS['vader']



    def __init__(self,
                 n_components: Optional[int] = None,
                 max_k: int = 8,
                 covariance_type: str = 'full',
                 random_state: int = 42,
                 features: Optional[List[str]] = None,
                 sentiment: str = 'vader'):

        # n_components: number of Gaussian components (clusters). None = auto-select via BIC
        # max_k: upper bound for component search when n_components is None
        # covariance_type: 'full' allows elliptical clusters, 'diag' for axis-aligned, 'spherical' like K-Means
        # for financial data 'full' is best because sentiment and returns are correlated

        self.n_components = n_components
        self.max_k = max_k
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.sentiment = sentiment
        self.features = features or self.FEATURE_SETS.get(sentiment, self.DEFAULT_FEATURES)
        self.scaler = StandardScaler()
        self.model = None
        self.fitted = False



        # store metrics after fitting
        self.bic_scores_ = {}           # BIC for each k tested
        self.aic_scores_ = {}           # AIC for each k tested
        self.best_k_ = None            # the k that was actually used
        self.evaluation_metrics_ = {}   # final model metrics


    def _find_optimal_k(self, X_scaled: np.ndarray) -> int:

        # find best number of components using BIC 
        # lower BIC = better balance of fit quality vs model complexity

        print(f"  Auto-selecting k via BIC (testing k=2..{self.max_k})...")

        best_k = 2
        best_bic = np.inf

        for k in range(2, self.max_k + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                n_init=5,           # run 5 times with different seeds, pick best
                max_iter=200,
            )
            gmm.fit(X_scaled)

            # bic + aic
            bic = gmm.bic(X_scaled)
            aic = gmm.aic(X_scaled)

            self.bic_scores_[k] = bic
            self.aic_scores_[k] = aic

            print(f"    k={k}: BIC={bic:.1f}, AIC={aic:.1f}")

            if bic < best_bic:
                best_bic = bic
                best_k = k


        print(f"  Best k={best_k} (BIC={best_bic:.1f})")
        return best_k


    def fit_predict(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:


        # main method - scales features, fits GMM, returns df with cluster labels
        # same interface as KMeansClusterer and SentimentClusterer
        # adds cluster_label and cluster_probability columns
        # cluster_probability = max probability from predict_proba (how confident the assignment is)

        df = feature_matrix.copy()

        # figure out which features we can actually use
        available_features = [f for f in self.features if f in df.columns]

        # error case - not enough features
        if len(available_features) < 2:
            print(f"ERROR: need at least 2 features, only found {available_features}")
            df['cluster_label'] = -1
            df['cluster_probability'] = 0.0
            return df
        
        # if available features = less than what we have default selected
        if len(available_features) < len(self.features):
            missing = set(self.features) - set(available_features)
            print(f"  heads up: missing features {missing}, using what we have")

        # show stats in terminal
        print(f"\nGMM clustering with {len(available_features)} features: {available_features}")
        print(f"  data points (trading days): {len(df)}")
        print(f"  covariance type: {self.covariance_type}")

        # grab feature columns and drop NaN rows
        X = df[available_features].copy()
        valid_mask = X.notna().all(axis=1)
        X_clean = X[valid_mask]

        # error case - not enough valid data points
        if len(X_clean) < 3:
            print(f"  not enough valid data points ({len(X_clean)}) for GMM")
            df['cluster_label'] = -1
            df['cluster_probability'] = 0.0
            return df

        dropped = len(X) - len(X_clean)
        if dropped > 0:
            print(f"  dropped {dropped} rows with NaN values")

        # scale features
        X_scaled = self.scaler.fit_transform(X_clean)

        # determine number of components
        if self.n_components is not None:
            k = self.n_components
            if k >= len(X_clean):
                k = max(2, len(X_clean) - 1)
                print(f"  adjusted k to {k} (not enough data for k={self.n_components})")
        else:
            # auto-select with BIC
            actual_max_k = min(self.max_k, len(X_clean) - 1)
            if actual_max_k < 2:
                actual_max_k = 2
            self.max_k = actual_max_k
            k = self._find_optimal_k(X_scaled)

        self.best_k_ = k

        # fit final model
        self.model = GaussianMixture(
            n_components=k,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=5,
            max_iter=200,
        )
        self.model.fit(X_scaled)

        # get hard labels and soft probabilities
        labels = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)

        # cluster_probability = how confident GMM is about the assignment
        # this is the max probability across all components for each day
        # so high value = clearly belongs to one cluster, low value = split between clusters
        max_probs = probs.max(axis=1)

        # calculate the evaluation metrics
        self.evaluation_metrics_ = {
            'silhouette': silhouette_score(X_scaled, labels),
            'calinski_harabasz': calinski_harabasz_score(X_scaled, labels),
            'davies_bouldin': davies_bouldin_score(X_scaled, labels),
            'bic': self.model.bic(X_scaled),
            'aic': self.model.aic(X_scaled),
        }

        # add results to df
        df['cluster_label'] = -1
        df['cluster_probability'] = 0.0
        df.loc[valid_mask, 'cluster_label'] = labels
        df.loc[valid_mask, 'cluster_probability'] = max_probs

        self.fitted = True
        self._print_summary(df)

        return df


    def _print_summary(self, df: pd.DataFrame):

        # print what the clustering found

        n_clusters = df[df['cluster_label'] >= 0]['cluster_label'].nunique()

        noise_count = (df['cluster_label'] == -1).sum()

        total = len(df)
        
        # before eval stats, print findings
        print(f"\n{'=' * 50}")
        print(f"GMM CLUSTERING RESULTS")
        print(f"{'=' * 50}")
        print(f"  components (k): {n_clusters}")
        print(f"  covariance:     {self.covariance_type}")
        print(f"  unassigned:     {noise_count}/{total} ({100*noise_count/total:.0f}%)")

        # print evaluation metrics
        if self.evaluation_metrics_:
            print(f"\n  Evaluation metrics:")
            print(f"    silhouette score:  {self.evaluation_metrics_['silhouette']:.4f}")
            print(f"    calinski-harabasz: {self.evaluation_metrics_['calinski_harabasz']:.2f}")
            print(f"    davies-bouldin:    {self.evaluation_metrics_['davies_bouldin']:.4f}")
            print(f"    BIC:               {self.evaluation_metrics_['bic']:.1f}")
            print(f"    AIC:               {self.evaluation_metrics_['aic']:.1f}")

        # show each clusters characteristics
        for label in sorted(df['cluster_label'].unique()):
            if label == -1:
                continue

            cluster_df = df[df['cluster_label'] == label]
            avg_prob = cluster_df['cluster_probability'].mean()
            print(f"\n  Cluster {label} ({len(cluster_df)} days, avg confidence={avg_prob:.3f}):")

            # print whichever sentiment columns exist
            for col in ['vader_mean', 'finbert_mean']:
                if col in cluster_df.columns:
                    print(f"    avg sentiment:  {cluster_df[col].mean():.4f}")
            if 'daily_return' in cluster_df.columns:
                print(f"    avg return:     {cluster_df['daily_return'].mean():.4f}")
            if 'realised_volatility_5d' in cluster_df.columns:
                print(f"    avg volatility: {cluster_df['realised_volatility_5d'].mean():.4f}")
            if 'positive_ratio' in cluster_df.columns:
                print(f"    pos/neg ratio:  {cluster_df['positive_ratio'].mean():.2f} / {cluster_df['negative_ratio'].mean():.2f}")


    def get_cluster_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        # same interface as the other clusterers
        if 'cluster_label' not in df.columns:
            print("no cluster labels found - run fit_predict first")
            return pd.DataFrame()

        clustered = df[df['cluster_label'] >= 0].copy()

        if clustered.empty:
            print("no clusters found")
            return pd.DataFrame()

        feature_cols = [f for f in self.features if f in df.columns]

        profiles = clustered.groupby('cluster_label')[feature_cols].agg(['mean', 'std'])
        counts = clustered.groupby('cluster_label').size().rename('day_count')

        profiles.columns = ['_'.join(col) for col in profiles.columns]
        profiles = profiles.join(counts)

        print(f"\nCluster profiles built for {len(profiles)} clusters")
        return profiles


# convenience function
def run_gmm_clustering(feature_matrix: pd.DataFrame,
                       n_components: Optional[int] = None,
                       max_k: int = 8,
                       random_state: int = 42) -> Tuple[pd.DataFrame, GMMClusterer]:
    clusterer = GMMClusterer(
        n_components=n_components,
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

        result, clusterer = run_gmm_clustering(df, max_k=6)
        profiles = clusterer.get_cluster_profiles(result)
        if not profiles.empty:
            print(f"\n{profiles.to_string()}")
    else:
        print(f"no feature matrix found at {test_path}")
        print("run the pipeline first: python pipeline.py --ticker NVDA --days 30")
