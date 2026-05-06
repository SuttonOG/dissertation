

# runs all 4 clustering methods with both sentiment scorers
# then generates all the chapter 4 visualisations
# 



# to use it: first run the p
# ipeline once to collect articles:
#   python pipeline.py --ticker NVDA --days 90 --scrape --rss --sentiment vader
# then run this script:
#   python run_comparison.py --ticker NVDA --days 90

import argparse
import os
import pandas as pd
import numpy as np
import time

from processing.sentiment_vader import VaderScorer
from processing.feature_aggregate import build_feature_matrix
from analysis.clustering import run_clustering
from analysis.statistical_validation import validate_clusters, save_validation_report


def run_comparison(ticker: str = "NVDA", days_back: int = 90, data_dir: str = "data"):

    start_time = time.time()



    # STEP 1: load the saved articles and prices 
    articles_path = os.path.join(data_dir, f"articles_{ticker}_{days_back}d.csv")
    prices_path = os.path.join(data_dir, f"prices_{ticker}_{days_back}d.csv")

    if not os.path.exists(articles_path):
        print(f"ERROR: cant find {articles_path}")
        print(f"run the pipeline first: python pipeline.py --ticker {ticker} --days {days_back} --scrape --rss")
        return

    print("=" * 60)
    print("LOADING SAVED DATA")
    print("=" * 60)

    articles_df = pd.read_csv(articles_path)
    print(f"  Loaded {len(articles_df)} articles from {articles_path}")

    price_df = None
    if os.path.exists(prices_path):
        price_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        print(f"  Loaded {len(price_df)} price days from {prices_path}")
    else:
        print(f"  WARNING: no price data found at {prices_path}")




    #  STEP 2: score with both scorers 
    print("\n" + "=" * 60)
    print("SCORING WITH BOTH SENTIMENT METHODS")
    print("=" * 60)

    # vader scoring
    if 'vader_compound' not in articles_df.columns:
        print("\n  Scoring with VADER...")
        vader = VaderScorer()
        articles_df = vader.score_dataframe(articles_df)
    else:
        print("  VADER scores already present - skipping")

    # finbert scoring
    if 'finbert_compound' not in articles_df.columns:
        print("\n  Scoring with FinBERT (this takes a while)...")
        from processing.sentiment_finbert import FinBertScorer
        finbert = FinBertScorer()
        articles_df = finbert.score_dataframe(articles_df)
    else:
        print("  FinBERT scores already present - skipping")



    # save the dual-scored articles so dont have to redo 


    dual_scored_path = os.path.join(data_dir, f"articles_dual_scored_{ticker}_{days_back}d.csv")
    articles_df.to_csv(dual_scored_path, index=False)
    print(f"  Saved dual-scored articles: {dual_scored_path}")



    # STEP 3: build feature matrices for both scorers 
    print("\n" + "=" * 60)
    print("BUILDING FEATURE MATRICES")
    print("=" * 60)

    vader_features = build_feature_matrix(articles_df, price_df, sentiment='vader')
    finbert_features = build_feature_matrix(articles_df, price_df, sentiment='finbert')

    # save them
    vader_features.to_csv(os.path.join(data_dir, f"features_vader_{ticker}_{days_back}d.csv"), index=False)
    finbert_features.to_csv(os.path.join(data_dir, f"features_finbert_{ticker}_{days_back}d.csv"), index=False)





    #  STEP 4: run all 4 methods on both feature matrices 
    methods = ['kmeans', 'hdbscan', 'gmm', 'hmm']
    scorers = ['vader', 'finbert']

    # store all results for chart generation
    all_clustered = {}          # key = 'method_scorer', value = clustered df
    all_validation = {}         # key = 'method_scorer', value = validation dict
    hmm_transition = None       # store HMM transition matrix for heatmp
    hmm_durations = None

    for scorer in scorers:
        feature_matrix = vader_features if scorer == 'vader' else finbert_features

        for method in methods:
            run_key = f"{method}_{scorer}"

            print("\n" + "=" * 60)
            print(f"RUNNING: {method.upper()} + {scorer.upper()}")
            print("=" * 60)

            # run clustering
            try:
                clustered_df, clusterer = run_clustering(
                    feature_matrix,
                    method=method,
                    sentiment=scorer,
                    min_cluster_size=max(3, len(feature_matrix) // 10),
                    min_samples=2,
                )
            except Exception as e:
                print(f"  ERROR running {run_key}: {e}")
                continue

            # save clustered output
            output_path = os.path.join(data_dir, f"clustered_{run_key}_{ticker}_{days_back}d.csv")
            clustered_df.to_csv(output_path, index=False)
            print(f"  Saved: {output_path}")

            all_clustered[run_key] = clustered_df

            # run validation
            print(f"\n  Validating {run_key}...")
            val_results = validate_clusters(clustered_df)
            all_validation[run_key] = val_results

            # save validation report
            val_path = os.path.join(data_dir, f"validation_{run_key}_{ticker}_{days_back}d.csv")
            save_validation_report(val_results, val_path)

            # save HMM-specific outputs (use vader version for charts)
            if method == 'hmm' and scorer == 'vader':
                if hasattr(clusterer, 'get_transition_matrix'):
                    hmm_transition = clusterer.get_transition_matrix()
                    if hmm_transition is not None:
                        trans_path = os.path.join(data_dir, f"transition_matrix_{ticker}_{days_back}d.csv")
                        hmm_transition.to_csv(trans_path)
                        print(f"  Transition matrix saved: {trans_path}")

                    hmm_durations = clusterer.get_expected_durations()
                    if hmm_durations is not None:
                        dur_path = os.path.join(data_dir, f"regime_durations_{ticker}_{days_back}d.csv")
                        hmm_durations.to_csv(dur_path, index=False)
                        print(f"  Regime durations saved: {dur_path}")

            # save cluster profiles
            profiles = clusterer.get_cluster_profiles(clustered_df)
            if not profiles.empty:
                prof_path = os.path.join(data_dir, f"profiles_{run_key}_{ticker}_{days_back}d.csv")
                profiles.to_csv(prof_path)





    #STEP 5: generate chapter 4 visualisations
    print("\n" + "=" * 60)
    print("GENERATING CHAPTER 4 CHARTS")
    print("=" * 60)

    from visualization.chapter4_visuals import generate_chapter4_charts

    chart_dir = os.path.join(data_dir, "chapter4_charts")

    # for the regime timeline chart, usevader results for all 4 methods
    regime_dict = {}
    for method in methods:
        key = f"{method}_vader"
        if key in all_clustered:
            regime_dict[method] = all_clustered[key]

    generate_chapter4_charts(
        results_dict=regime_dict,
        validation_dict=all_validation,
        articles_df=articles_df,
        price_df=price_df,
        hmm_transition_df=hmm_transition,
        ticker=ticker,
        output_dir=chart_dir,
    )





    # STEP 6: print summary table
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    print(f"\n  {'Method':<20s} {'Clusters':>8s} {'KW p (ret)':>12s} {'KW p (vol)':>12s} "
          f"{'ε² (ret)':>10s} {'Avg |δ| ret':>12s}")
    print("  " + "─" * 74)

    for run_key in sorted(all_validation.keys()):
        val = all_validation[run_key]
        n_cl = val.get('n_clusters', 0)

        # get return metrics
        ret_data = val.get('metrics', {}).get('daily_return', {})
        kw_ret = ret_data.get('kruskal_wallis', {})
        ret_p = kw_ret.get('p_value', 1.0)
        ret_eps = kw_ret.get('epsilon_squared', 0)

        # avg cliff's delta for returns
        pairwise_ret = ret_data.get('pairwise', [])
        if pairwise_ret:
            avg_delta = np.mean([abs(p.get('cliffs_delta', 0)) for p in pairwise_ret])
        else:
            avg_delta = 0

        # get volatility stats
        vol_data = val.get('metrics', {}).get('realised_volatility_5d', {})
        kw_vol = vol_data.get('kruskal_wallis', {})
        vol_p = kw_vol.get('p_value', 1.0)

        sig_ret = "*" if ret_p < 0.05 else ""
        sig_vol = "*" if vol_p < 0.05 else ""

        print(f"  {run_key:<20s} {n_cl:>8d} {ret_p:>11.6f}{sig_ret} {vol_p:>11.6f}{sig_vol} "
              f"{ret_eps:>10.4f} {avg_delta:>12.4f}")

    print(f"\n  * = significant at α = 0.05")

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f} seconds")
    print(f"  All outputs saved to: {os.path.abspath(data_dir)}/")
    print(f"  Chapter 4 charts in: {os.path.abspath(chart_dir)}/")
    print("=" * 60)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run all clustering methods and generate comparison charts")
    parser.add_argument("--ticker", type=str, default="NVDA", help="Stock ticker (default: NVDA)")
    parser.add_argument("--days", type=int, default=90, help="Days of history (default: 90)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory (default: data)")

    args = parser.parse_args()

    run_comparison(
        ticker=args.ticker,
        days_back=args.days,
        data_dir=args.data_dir,
    )
