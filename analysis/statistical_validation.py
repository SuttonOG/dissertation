# statistical_validation.py
# tests whether the clusters we found actually have different market behaviour
# uses non-parametric tests because financial returns arent normally distributed
# three layers: kruskal-wallis (any different?), mann-whitney (which pairs?), cliffs delta (how much?)

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations


def cliffs_delta(group_x, group_y):
    # computes cliff's delta effect size between two groups
    # counts how often values in x are bigger than values in y
    # ranges from -1 to 1, where 0 means no difference
    # benchmarks: <0.147 negligible, <0.33 small, <0.474 medium, >=0.474 large

    nx = len(group_x)
    ny = len(group_y)

    if nx == 0 or ny == 0:
        return 0.0

    # count wins and losses
    more = 0
    less = 0
    for xi in group_x:
        for yi in group_y:
            if xi > yi:
                more += 1
            elif xi < yi:
                less += 1

    delta = (more - less) / (nx * ny)
    return delta


def interpret_cliffs_delta(d):
    # turn the number into a word
    d_abs = abs(d)
    if d_abs < 0.147:
        return "negligible"
    elif d_abs < 0.33:
        return "small"
    elif d_abs < 0.474:
        return "medium"
    else:
        return "large"


def validate_clusters(df, cluster_col='cluster_label', metrics=None, alpha=0.05):
    # runs the full validation on the clustered data
    # tests whether clusters have statistically different return and volatility distributions
    #
    # df: the feature matrix with cluster labels already assigned
    # cluster_col: which column has the cluster assignments
    # metrics: which columns to test (defaults to daily_return and volatility)
    # alpha: significance threshold
    #
    # returns a dict with all the test results

    if metrics is None:
        metrics = ['daily_return', 'realised_volatility_5d']

    # only look at assigned points, ignore noise (-1)
    clustered_df = df[df[cluster_col] >= 0].copy()
    cluster_ids = sorted(clustered_df[cluster_col].unique())
    num_clusters = len(cluster_ids)

    if num_clusters < 2:
        print("  Need at least 2 clusters to validate - skipping")
        return {}

    #  store everything here
    results = {
        'n_clusters': num_clusters,
        'cluster_sizes': {},
        'metrics': {},
    }

    # count how many days in each cluster
    for c in cluster_ids:
        results['cluster_sizes'][int(c)] = int((clustered_df[cluster_col] == c).sum())

    print(f"\n{'=' * 70}")
    print(f"STATISTICAL VALIDATION REPORT")
    print(f"{'=' * 70}")
    print(f"  Clusters: {num_clusters} (sizes: {results['cluster_sizes']})")
    print(f"  Metrics:  {metrics}")
    print(f"  Alpha:    {alpha}")

    # run validation for each metric (returns, volatility, etc)
    for metric in metrics:

        if metric not in clustered_df.columns:
            print(f"\n  WARNING: '{metric}' not in dataframe, skipping")
            continue

        print(f"\n{'─' * 70}")
        print(f"  METRIC: {metric}")
        print(f"{'─' * 70}")

        metric_results = {
            'descriptive': {},
            'kruskal_wallis': {},
            'pairwise': [],
        }

        # --- descriptive stats for each cluster ---
        print(f"\n  Descriptive stats:")
        for c in cluster_ids:
            cluster_data = clustered_df[clustered_df[cluster_col] == c][metric]
            desc = {
                'n': len(cluster_data),
                'median': float(cluster_data.median()),
                'mean': float(cluster_data.mean()),
                'std': float(cluster_data.std()),
            }
            metric_results['descriptive'][int(c)] = desc
            print(f"    Cluster {c}: n={desc['n']}, median={desc['median']:.5f}, "
                  f"mean={desc['mean']:.5f}, std={desc['std']:.5f}")

        # --- kruskal-wallis omnibus test ---
        # tests if ANY cluster is different from the others
        # non-parametric version of one-way ANOVA
        groups = [clustered_df[clustered_df[cluster_col] == c][metric].values for c in cluster_ids]
        h_stat, kw_pvalue = stats.kruskal(*groups)

        # epsilon squared effect size for kruskal-wallis
        # tells us what proportion of variance is explained by cluster membership
        n_total = len(clustered_df)
        epsilon_sq = (h_stat - num_clusters + 1) / (n_total - num_clusters)
        epsilon_sq = max(0, epsilon_sq)     # cant go negative

        # interpret the effect size
        if epsilon_sq < 0.06:
            eps_interpretation = 'small'
        elif epsilon_sq < 0.14:
            eps_interpretation = 'medium'
        else:
            eps_interpretation = 'large'

        metric_results['kruskal_wallis'] = {
            'H_statistic': float(h_stat),
            'p_value': float(kw_pvalue),
            'epsilon_squared': float(epsilon_sq),
            'effect_interpretation': eps_interpretation,
            'significant': bool(kw_pvalue < alpha),
        }

        print(f"\n  Kruskal-Wallis omnibus test:")
        print(f"    H = {h_stat:.4f}, p = {kw_pvalue:.6f}, "
              f"epsilon² = {epsilon_sq:.4f} ({eps_interpretation})")

        if kw_pvalue >= alpha:
            print(f"    → FAIL TO REJECT H0: clusters are not significantly different")
            results['metrics'][metric] = metric_results
            continue

        print(f"    → REJECT H0: at least one cluster differs (p < {alpha})")

        # --- pairwise mann-whitney U tests ---
        # now we know SOMETHING is different, lets find out WHICH pairs
        all_pairs = list(combinations(cluster_ids, 2))
        raw_pvalues = []
        pair_results_list = []

        for c1, c2 in all_pairs:
            data1 = clustered_df[clustered_df[cluster_col] == c1][metric].values
            data2 = clustered_df[clustered_df[cluster_col] == c2][metric].values

            u_stat, mw_pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')

            # cliff's delta for this pair
            cd = cliffs_delta(data1, data2)

            raw_pvalues.append(mw_pvalue)
            pair_results_list.append({
                'cluster_1': int(c1),
                'cluster_2': int(c2),
                'U_statistic': float(u_stat),
                'raw_p_value': float(mw_pvalue),
                'cliffs_delta': float(cd),
            })

        # bonferroni correction - multiply each p-value by number of tests
        # this controls the family-wise error rate when doing multiple comparisons
        num_tests = len(raw_pvalues)
        corrected_pvalues = [min(p * num_tests, 1.0) for p in raw_pvalues]

        # print the results table
        print(f"\n  Pairwise Mann-Whitney U (Bonferroni corrected, {num_tests} tests):")
        print(f"    {'Pair':20s} {'U':>8s} {'Raw p':>10s} {'Adj. p':>10s} {'Sig.':>5s} "
              f"{'Cliff d':>8s} {'Effect':>10s}")
        print(f"    {'─' * 71}")

        for i, pair_result in enumerate(pair_results_list):
            pair_result['corrected_p_value'] = corrected_pvalues[i]
            pair_result['significant'] = bool(corrected_pvalues[i] < alpha)
            pair_result['effect_interpretation'] = interpret_cliffs_delta(pair_result['cliffs_delta'])

            sig_marker = "*" if pair_result['significant'] else ""
            pair_name = f"C{pair_result['cluster_1']} vs C{pair_result['cluster_2']}"
            print(f"    {pair_name:20s} {pair_result['U_statistic']:8.0f} "
                  f"{pair_result['raw_p_value']:10.6f} {pair_result['corrected_p_value']:10.6f} "
                  f"{sig_marker:>5s} {pair_result['cliffs_delta']:8.4f} "
                  f"{pair_result['effect_interpretation']:>10s}")

        metric_results['pairwise'] = pair_results_list
        results['metrics'][metric] = metric_results

    print(f"\n{'=' * 70}")
    print(f"  * = significant after Bonferroni correction (alpha = {alpha})")
    print(f"  Cliff's delta: negligible < 0.147, small < 0.33, medium < 0.474, large >= 0.474")
    print(f"{'=' * 70}")

    return results


def save_validation_report(results, output_path):
    # saves the validation results as a csv so we can use them later

    rows = []

    for metric, metric_data in results.get('metrics', {}).items():

        # add the kruskal-wallis row
        kw = metric_data.get('kruskal_wallis', {})
        rows.append({
            'metric': metric,
            'test': 'Kruskal-Wallis',
            'comparison': 'omnibus',
            'statistic': kw.get('H_statistic'),
            'p_value': kw.get('p_value'),
            'corrected_p_value': kw.get('p_value'),
            'effect_size': kw.get('epsilon_squared'),
            'effect_type': 'epsilon_squared',
            'effect_interpretation': kw.get('effect_interpretation'),
            'significant': kw.get('significant'),
        })

        # add each pairwise test row
        for pw in metric_data.get('pairwise', []):
            rows.append({
                'metric': metric,
                'test': 'Mann-Whitney U',
                'comparison': f"C{pw['cluster_1']} vs C{pw['cluster_2']}",
                'statistic': pw.get('U_statistic'),
                'p_value': pw.get('raw_p_value'),
                'corrected_p_value': pw.get('corrected_p_value'),
                'effect_size': pw.get('cliffs_delta'),
                'effect_type': 'cliffs_delta',
                'effect_interpretation': pw.get('effect_interpretation'),
                'significant': pw.get('significant'),
            })

    if rows:
        report_df = pd.DataFrame(rows)
        report_df.to_csv(output_path, index=False)
        print(f"  Validation report saved: {output_path}")
    else:
        print("  No validation results to save")


# quick test
if __name__ == "__main__":

    np.random.seed(42)

    # make some fake clustered data with 3 obviously different groups
    test_df = pd.DataFrame({
        'cluster_label': [0]*40 + [1]*30 + [2]*35,
        'daily_return': np.concatenate([
            np.random.normal(0.008, 0.006, 40),     # cluster 0: positive
            np.random.normal(-0.008, 0.010, 30),    # cluster 1: negative
            np.random.normal(0.001, 0.005, 35),     # cluster 2: flat
        ]),
        'realised_volatility_5d': np.concatenate([
            np.random.normal(0.009, 0.003, 40),     # cluster 0: low vol
            np.random.normal(0.022, 0.005, 30),     # cluster 1: high vol
            np.random.normal(0.014, 0.004, 35),     # cluster 2: medium vol
        ]),
    })

    print("Running validation on test data...")
    test_results = validate_clusters(test_df)
    save_validation_report(test_results, "data/validation_test.csv")
    print("\nDone!")
