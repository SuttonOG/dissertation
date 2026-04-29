 

 # Hidden Markov Model clustering on the feature matrix
# apparently the gold standard for financial regime detection
# unlike the other models, HMM cares about the TIME ORDER of the data
# learns transition probabilities between regimes (e.g. stressed -> neutral is more likely than stressed -> optimistic)


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from hmmlearn.hmm import GaussianHMM
from typing import Optional, List, Tuple
import warnings




class HMMClusterer:

    # HMM clustering on the daily feature matrix
    # KEY DIFFERENCE from K-Means/GMM/HDBSCAN === data must be in TIME ORDER
    # the model learns:
    #  transition probabilities (how likely to switch from regime A to B)
    # emission parameters (what each regime's data looks like)
    #   initial state probabilities (which regime does the sequence start in)




    # same feature sets as other clusterers for fair comparison
    
    # select features to use based on sentiment used vader/finbert
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
                 max_states: int = 6,
                 covariance_type: str = 'full',
                 n_iter: int = 200,
                 n_random_inits: int = 5,
                 random_state: int = 42,
                 features: Optional[List[str]] = None,
                 sentiment: str = 'vader'):


            # Args:
        # n_components: number of hidden states (regimes). None = auto-select via BIC
        # max_states: upper bound for state search when n_components is None
        # n_random_inits: how many times to refit with different random seeds
        #   HMM's EM algorithm gets stuck in local optima easily, so we run it
        #   multiple times and pick the best result
        # n_iter: max EM iterations per fit attempt

        self.n_components = n_components
        self.max_states = max_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.n_random_inits = n_random_inits
        self.random_state = random_state
        self.sentiment = sentiment
        self.features = features or self.FEATURE_SETS.get(sentiment, self.DEFAULT_FEATURES)
        self.scaler = StandardScaler()
        self.model = None
        self.fitted = False

        # store results
        self.bic_scores_ = {}
        self.best_k_ = None
        self.transition_matrix_ = None     # the learned transition probabilities
        self.evaluation_metrics_ = {}


    def _compute_bic(self, model, X):
        # compute BIC for a fitted HMM
        # BIC = -2 * log_likelihood + n_params * log(n_observations)
        # lower is better

        K = model.n_components
        T, d = X.shape
        log_likelihood = model.score(X)

        # count free parameters in the model:
        # start probabilities: K - 1 (they sum to 1)
        # transition matrix: K * (K - 1) (each row sums to 1)
        # emission means: K * d
        # emission covariances (full): K * d * (d + 1) / 2
        n_params = (K - 1) + K * (K - 1) + K * d + K * d * (d + 1) // 2

        bic = -2 * log_likelihood + n_params * np.log(T)
        return bic


    def _fit_single(self, X_scaled, n_components):
        # fit a single HMM with multiple random initialisations
        # return the best model (highest log-likelihood)

        best_model = None
        best_score = -np.inf

        for seed in range(self.n_random_inits):
            try:
                # suppress convergence warnings - handle non-convergence personally
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    model = GaussianHMM(
                        n_components=n_components,
                        covariance_type=self.covariance_type,
                        n_iter=self.n_iter,
                        random_state=self.random_state + seed,
                        verbose=False,
                    )
                    model.fit(X_scaled)

                    score = model.score(X_scaled)
                    if score > best_score:
                        best_score = score
                        best_model = model

            except Exception as e:
                # some initialisations fail (singular covariance etc) - thats ok, skip
                continue

        return best_model


    def _find_optimal_states(self, X_scaled: np.ndarray) -> int:
        # find best number of hidden states using BIC
        print(f"  Auto-selecting states via BIC (testing n=2..{self.max_states})...")

        best_k = 2
        best_bic = np.inf

        for n_states in range(2, self.max_states + 1):
            model = self._fit_single(X_scaled, n_states)

            if model is None:
                print(f"    n={n_states}: FAILED to fit (all initialisations failed)")
                continue

            bic = self._compute_bic(model, X_scaled)
            log_lik = model.score(X_scaled)
            self.bic_scores_[n_states] = bic

            print(f"    n={n_states}: BIC={bic:.0f}, LogLik={log_lik:.0f}")

            if bic < best_bic:
                best_bic = bic
                best_k = n_states

        print(f"  Best n={best_k} (BIC={best_bic:.0f})")
        return best_k


    def fit_predict(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:

        # main method !!!!!! IMPORTANT!!!!!!! : data must be sorted by date before calling fit_predictS
        # sorts by published_day or trading_day if avail
        # fits GaussianHMM, predicts hidden state sequence using Viterbi algorithm
        # returns df with cluster_label and cluster_probability

        df = feature_matrix.copy()

        # CRITICAL: sort by date - HMM needs time order
        if 'trading_day' in df.columns:
            df = df.sort_values('trading_day').reset_index(drop=True)
        elif 'published_day' in df.columns:
            df = df.sort_values('published_day').reset_index(drop=True)

        # figure out which features we can use
        available_features = [f for f in self.features if f in df.columns]

        if len(available_features) < 2:
            print(f"ERROR: need at least 2 features, only found {available_features}")
            df['cluster_label'] = -1
            df['cluster_probability'] = 0.0
            return df

        if len(available_features) < len(self.features):
            missing = set(self.features) - set(available_features)
            print(f"  heads up: missing features {missing}, using what we have")

        print(f"\nHMM clustering with {len(available_features)} features: {available_features}")
        print(f"  data points (trading days): {len(df)}")
        print(f"  covariance type: {self.covariance_type}")
        print(f"  random initialisations per model: {self.n_random_inits}")

        # grab feature columns and drop NaN rows
        X = df[available_features].copy()
        valid_mask = X.notna().all(axis=1)
        X_clean = X[valid_mask]

        if len(X_clean) < 5:
            print(f"  not enough valid data points ({len(X_clean)}) for HMM")
            df['cluster_label'] = -1
            df['cluster_probability'] = 0.0
            return df

        dropped = len(X) - len(X_clean)
        if dropped > 0:
            print(f"  dropped {dropped} rows with NaN values")

        # scale features
        X_scaled = self.scaler.fit_transform(X_clean)

        # determine number of hidden states
        if self.n_components is not None:
            k = self.n_components
            if k >= len(X_clean):
                k = max(2, len(X_clean) - 1)
                print(f"  adjusted k to {k} (not enough data for k={self.n_components})")
        else:
            actual_max = min(self.max_states, len(X_clean) // 3)  # need at least 3 points per state
            if actual_max < 2:
                actual_max = 2
            self.max_states = actual_max
            k = self._find_optimal_states(X_scaled)

        self.best_k_ = k

        # fit final model with best k
        print(f"  Fitting final model with {k} states...")
        self.model = self._fit_single(X_scaled, k)

        if self.model is None:
            print("  ERROR: HMM failed to fit. Returning all as noise.")
            df['cluster_label'] = -1
            df['cluster_probability'] = 0.0
            return df

        # predict hidden states using Viterbi algorithm
        # this finds the most likely STATE SEQUENCE (not just individual assignments)
        labels = self.model.predict(X_scaled)

        # get state probabilities for each day
        state_probs = self.model.predict_proba(X_scaled)
        max_probs = state_probs.max(axis=1)

        # store the transition matrix - the most important HMM output
        self.transition_matrix_ = self.model.transmat_

        # compute evaluation metrics (same as other clusterers for comparison)
        self.evaluation_metrics_ = {
            'silhouette': silhouette_score(X_scaled, labels),
            'calinski_harabasz': calinski_harabasz_score(X_scaled, labels),
            'davies_bouldin': davies_bouldin_score(X_scaled, labels),
            'log_likelihood': self.model.score(X_scaled),
            'bic': self._compute_bic(self.model, X_scaled),
            'converged': self.model.monitor_.converged,
        }

        # add results to dataframe
        df['cluster_label'] = -1
        df['cluster_probability'] = 0.0
        df.loc[valid_mask, 'cluster_label'] = labels
        df.loc[valid_mask, 'cluster_probability'] = max_probs

        self.fitted = True
        self._print_summary(df)

        return df


    def _print_summary(self, df: pd.DataFrame):
        # print clustering results

        n_clusters = df[df['cluster_label'] >= 0]['cluster_label'].nunique()

        noise_count = (df['cluster_label'] == -1).sum()

        total = len(df)



        print(f"\n{'=' * 50}")
        print(f"HMM CLUSTERING RESULTS")
        print(f"{'=' * 50}")
        print(f"  hidden states:  {n_clusters}")
        print(f"  unassigned:     {noise_count}/{total} ({100*noise_count/total:.0f}%)")



        # print eval stats if exists
        if self.evaluation_metrics_:
            print(f"\n  Evaluation metrics:")
            print(f"    silhouette score:  {self.evaluation_metrics_['silhouette']:.4f}")               # silhouette
            print(f"    calinski-harabasz: {self.evaluation_metrics_['calinski_harabasz']:.2f}")            # calinski

            print(f"    davies-bouldin:    {self.evaluation_metrics_['davies_bouldin']:.4f}")               # davies bouldin
            print(f"    log-likelihood:    {self.evaluation_metrics_['log_likelihood']:.2f}")               # log likelihood
            print(f"    BIC:               {self.evaluation_metrics_['bic']:.0f}")                          # BIC
            print(f"    converged:         {self.evaluation_metrics_['converged']}")                        # converged



        # print transition matrix
        if self.transition_matrix_ is not None:
            print(f"\n  Transition matrix (row=from, col=to):")
            k = self.transition_matrix_.shape[0]
            header = "        " + "  ".join([f"  State {j}" for j in range(k)])
            print(header)
            for i in range(k):
                row = "  ".join([f"{self.transition_matrix_[i, j]:8.4f}" for j in range(k)])
                print(f"    S{i}:  {row}")

            # expected regime durations
            print(f"\n  Expected regime durations:")
            for i in range(k):
                self_trans = self.transition_matrix_[i, i]
                if self_trans < 1.0:
                    duration = 1.0 / (1.0 - self_trans)
                else:
                    duration = float('inf')
                print(f"    State {i}: {duration:.1f} days (self-transition={self_trans:.4f})")



        # show each states characteristics
        for label in sorted(df['cluster_label'].unique()):
            if label == -1:
                continue

            cluster_df = df[df['cluster_label'] == label]
            print(f"\n  State {label} ({len(cluster_df)} days):")

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
        # same interface as other clusterers
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




    def get_transition_matrix(self) -> Optional[pd.DataFrame]:
        # return the transition matrix as a readable dataframe
        # this is the unique HMM output that other methods cant produce
        if self.transition_matrix_ is None:
            print("no transition matrix - run fit_predict first")
            return None

        k = self.transition_matrix_.shape[0]
        labels = [f"State {i}" for i in range(k)]
        return pd.DataFrame(self.transition_matrix_, index=labels, columns=labels).round(4)




    def get_expected_durations(self) -> Optional[pd.DataFrame]:
        # compute expected duration of each regime from the transition matrix
        # expected duration = 1 / (1 - self_transition_probability)
        if self.transition_matrix_ is None:
            print("no transition matrix - run fit_predict first")
            return None

        k = self.transition_matrix_.shape[0]
        durations = []
        for i in range(k):
            self_trans = self.transition_matrix_[i, i]
            if self_trans < 1.0:
                duration = 1.0 / (1.0 - self_trans)
            else:
                duration = float('inf')
            durations.append({
                'state': i,
                'self_transition': round(self_trans, 4),
                'expected_duration_days': round(duration, 1),
            })

        return pd.DataFrame(durations)




# convenience function
def run_hmm_clustering(feature_matrix: pd.DataFrame,
                       n_components: Optional[int] = None,
                       max_states: int = 6,
                       random_state: int = 42) -> Tuple[pd.DataFrame, HMMClusterer]:
    clusterer = HMMClusterer(
        n_components=n_components,
        max_states=max_states,
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

        result, clusterer = run_hmm_clustering(df, max_states=5)

        # print profiles
        profiles = clusterer.get_cluster_profiles(result)
        if not profiles.empty:
            print(f"\n{profiles.to_string()}")

        # print transition matrix
        trans = clusterer.get_transition_matrix()
        if trans is not None:
            print(f"\nTransition Matrix:")
            print(trans.to_string())

        # print expected durations
        durations = clusterer.get_expected_durations()
        if durations is not None:
            print(f"\nExpected Durations:")
            print(durations.to_string(index=False))
    else:
        print(f"no feature matrix found at {test_path}")
        print("run the pipeline first: python pipeline.py --ticker NVDA --days 30")
