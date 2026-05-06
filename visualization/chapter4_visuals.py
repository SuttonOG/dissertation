# chapter4_visuals.py
# visualisations for the results chapter of diss
# designed to address the 4 project goals (G1-G4)



# run AFTER pipeline has been executed for all method/scorer combos


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional
import os




# same style as the main visuals file
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})









# CHART 1: VADER vs FinBERT scatter (section 4.2)
# shows how the two scorers rate the same articles


def plot_scorer_comparison(articles_df: pd.DataFrame, ticker: str = "",
                           output_dir: str = "output/charts"):
    # scatter plot comparing vader and finbert scores on the same articles
    # needs a df that has BOTH vader_compound and finbert_compound columns
    # run both scorers on the same articles first

    os.makedirs(output_dir, exist_ok=True)

    if 'vader_compound' not in articles_df.columns or 'finbert_compound' not in articles_df.columns:
        print("  need both vader_compound and finbert_compound columns - skipping")
        return None

    fig, ax = plt.subplots(figsize=(9, 9))

    vader_scores = articles_df['vader_compound'].dropna()
    finbert_scores = articles_df['finbert_compound'].dropna()

    # only plot articles that have both scores
    both_mask = articles_df['vader_compound'].notna() & articles_df['finbert_compound'].notna()
    v = articles_df.loc[both_mask, 'vader_compound']
    f = articles_df.loc[both_mask, 'finbert_compound']

    ax.scatter(v, f, alpha=0.3, s=20, color='#2196F3', edgecolors='none')

    # diagonal line = perfect agreement
    ax.plot([-1, 1], [-1, 1], '--', color='gray', linewidth=1, alpha=0.5, label='Perfect agreement')

    # correlation
    if len(v) > 2:
        corr = v.corr(f)
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}\nn = {len(v)} articles',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('VADER Compound Score')
    ax.set_ylabel('FinBERT Compound Score')
    ax.set_title(f'VADER vs FinBERT Scoring Comparison{" - " + ticker if ticker else ""}')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc='lower right')
    plt.tight_layout()

    path = os.path.join(output_dir, 'scorer_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")
    return path






# CHART 2: Sentiment distribution comparison (section 4.2)
# side by side histograms for vader vs finbert


def plot_scorer_distributions(articles_df: pd.DataFrame, ticker: str = "",
                               output_dir: str = "output/charts"):
    # side by side histograms showing how each scorer distributes scores

    os.makedirs(output_dir, exist_ok=True)

    if 'vader_compound' not in articles_df.columns or 'finbert_compound' not in articles_df.columns:
        print("  need both scorer columns for distribution comparison - skipping")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)


    # vader histogram
    vader_vals = articles_df['vader_compound'].dropna()
    ax1.hist(vader_vals, bins=40, color='#2196F3', alpha=0.7, edgecolor='white')
    ax1.axvline(x=vader_vals.mean(), color='#F44336', linestyle='--', linewidth=2,
                label=f'Mean = {vader_vals.mean():.3f}')
    ax1.set_xlabel('Compound Score')
    ax1.set_ylabel('Number of Articles')
    ax1.set_title('VADER Distribution')
    ax1.legend()



    # finbert histogram
    finbert_vals = articles_df['finbert_compound'].dropna()
    ax2.hist(finbert_vals, bins=40, color='#4CAF50', alpha=0.7, edgecolor='white')
    ax2.axvline(x=finbert_vals.mean(), color='#F44336', linestyle='--', linewidth=2,
                label=f'Mean = {finbert_vals.mean():.3f}')
    ax2.set_xlabel('Compound Score')
    ax2.set_title('FinBERT Distribution')
    ax2.legend()



    fig.suptitle(f'Sentiment Score Distributions{" - " + ticker if ticker else ""}', fontsize=15)
    plt.tight_layout()

    path = os.path.join(output_dir, 'scorer_distributions.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")
    return path







# CHART 3: Regime timelines stacked (section 4.3)
# shows all 4 methods' regime assignments as colour bars


def plot_regime_timelines(results_dict: Dict[str, pd.DataFrame],
                          ticker: str = "", output_dir: str = "output/charts"):
    # stacked timeline showing regime assignments for each method
    # results_dict = {'kmeans': df_with_cluster_label, 'hdbscan': df, 'gmm': df, 'hmm': df}
    # each df must have 'published_day' or 'trading_day' and 'cluster_label'

    os.makedirs(output_dir, exist_ok=True)

    methods = list(results_dict.keys())
    n_methods = len(methods)

    if n_methods == 0:
        print("  no results to plot regime timelines - skipping")
        return None

    fig, axes = plt.subplots(n_methods, 1, figsize=(16, 2.5 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]

    palette = sns.color_palette("deep", n_colors=10)
    noise_colour = 'lightgray'

    for idx, method_name in enumerate(methods):
        ax = axes[idx]
        df = results_dict[method_name]

        # figure out the date column
        if 'trading_day' in df.columns:
            dates = pd.to_datetime(df['trading_day'])
        elif 'published_day' in df.columns:
            dates = pd.to_datetime(df['published_day'])
        else:
            print(f"  no date column found for {method_name} - skipping")
            continue

        labels = df['cluster_label'].values

        # plot each day as a coloured bar
        for i in range(len(dates)):
            lbl = int(labels[i])
            if lbl == -1:
                colour = noise_colour
            else:
                colour = palette[lbl % len(palette)]
            ax.axvspan(dates.iloc[i] - pd.Timedelta(hours=12),
                       dates.iloc[i] + pd.Timedelta(hours=12),
                       color=colour, alpha=0.8)

        ax.set_ylabel(method_name.upper(), fontsize=12, fontweight='bold')
        ax.set_yticks([])

        # add legend for this methods clusters
        unique_labels = sorted(df[df['cluster_label'] >= 0]['cluster_label'].unique())
        legend_handles = []
        for lbl in unique_labels:
            patch = plt.Rectangle((0, 0), 1, 1, fc=palette[lbl % len(palette)])
            legend_handles.append(patch)
        legend_labels = [f'C{l}' for l in unique_labels]

        if (df['cluster_label'] == -1).any():
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=noise_colour))
            legend_labels.append('Noise')

        ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=8, ncol=len(legend_labels))

    axes[-1].set_xlabel('Date')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.xticks(rotation=45)
    fig.suptitle(f'Regime Assignments by Method{" - " + ticker if ticker else ""}',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'regime_timelines.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")
    return path



# CHART 4: Return distributions by cluster 
# box/violin plots showing return distribution within each cluster
# directly visualises what the kruskal-wallis test is testing



def plot_return_distributions(df: pd.DataFrame, method_name: str = "",
                               ticker: str = "", output_dir: str = "output/charts"):
    # box plots of daily returns grouped by cluster
    # if the boxes dont overlap -> clusters have different return behaviour

    os.makedirs(output_dir, exist_ok=True)

    if 'cluster_label' not in df.columns or 'daily_return' not in df.columns:
        print("  need cluster_label and daily_return columns - skipping")
        return None

    # only assigned points
    clustered = df[df['cluster_label'] >= 0].copy()
    if clustered.empty:
        print("  no clusters found - skipping return distributions")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    palette = sns.color_palette("deep", n_colors=clustered['cluster_label'].nunique())



    # left: returns by cluster
    cluster_ids = sorted(clustered['cluster_label'].unique())
    returns_data = [clustered[clustered['cluster_label'] == c]['daily_return'].values for c in cluster_ids]

    bp1 = ax1.boxplot(returns_data, tick_labels=[f'C{c}' for c in cluster_ids], patch_artist=True)
    for i, patch in enumerate(bp1['boxes']):
        patch.set_facecolor(palette[i % len(palette)])
        patch.set_alpha(0.6)
    ax1.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Daily Return')
    ax1.set_title('Daily Returns by Cluster')





    # right: volatility by cluster
    if 'realised_volatility_5d' in clustered.columns:
        vol_data = [clustered[clustered['cluster_label'] == c]['realised_volatility_5d'].values for c in cluster_ids]
        bp2 = ax2.boxplot(vol_data, tick_labels=[f'C{c}' for c in cluster_ids], patch_artist=True)
        for i, patch in enumerate(bp2['boxes']):
            patch.set_facecolor(palette[i % len(palette)])
            patch.set_alpha(0.6)
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Realised Volatility (5d)')
        ax2.set_title('Volatility by Cluster')

    title_suffix = f" ({method_name.upper()})" if method_name else ""
    fig.suptitle(f'Return and Volatility Distributions{title_suffix}{" - " + ticker if ticker else ""}',
                 fontsize=15)
    plt.tight_layout()

    fname = f'return_distributions_{method_name}.png' if method_name else 'return_distributions.png'
    path = os.path.join(output_dir, fname)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")
    return path





# CHART 5: HMM transition matrix heatmap (section 4.3)


def plot_transition_matrix(transition_df: pd.DataFrame, ticker: str = "",
                            output_dir: str = "output/charts"):
    # heatmap of the HMM transition probability matrix
    # transition_df should be the output of clusterer.get_transition_matrix()

    os.makedirs(output_dir, exist_ok=True)

    if transition_df is None or transition_df.empty:
        print("  no transition matrix to plot - skipping")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(transition_df.values.astype(float), annot=True, fmt='.3f',
                cmap='YlOrRd', vmin=0, vmax=1,
                xticklabels=transition_df.columns,
                yticklabels=transition_df.index,
                ax=ax, linewidths=0.5, linecolor='white')

    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title(f'HMM Transition Probability Matrix{" - " + ticker if ticker else ""}')
    plt.tight_layout()

    path = os.path.join(output_dir, 'transition_matrix_heatmap.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")
    return path





# CHART 6: Model comparison bar chart (section 4.4)
# grouped bars showing effect sizes across all methods


def plot_model_comparison(validation_results: Dict[str, Dict],
                           output_dir: str = "output/charts"):
    # grouped bar chart comparing avg cliff's delta across methods
    # validation_results = {'kmeans_vader': validate_clusters_output, ...}
    # each value is the dict returned by validate_clusters()

    os.makedirs(output_dir, exist_ok=True)

    if not validation_results:
        print("  no validation results to compare - skipping")
        return None

    # extract avg absolute cliff's delta for each method x metric
    rows = []
    for method_key, val_result in validation_results.items():
        for metric, metric_data in val_result.get('metrics', {}).items():
            # get the kruskal wallis effect size
            kw = metric_data.get('kruskal_wallis', {})

            # get avg absolute cliff's delta from pairwise tests
            pairwise = metric_data.get('pairwise', [])
            if pairwise:
                avg_cliff = np.mean([abs(p.get('cliffs_delta', 0)) for p in pairwise])
            else:
                avg_cliff = 0

            rows.append({
                'method': method_key,
                'metric': metric,
                'kw_h': kw.get('H_statistic', 0),
                'kw_p': kw.get('p_value', 1),
                'epsilon_sq': kw.get('epsilon_squared', 0),
                'avg_cliffs_delta': avg_cliff,
                'significant': kw.get('significant', False),
            })

    if not rows:
        print("  no comparison data extracted - skipping")
        return None

    comp_df = pd.DataFrame(rows)

    # plot avg cliff's delta grouped by metric
    metrics_list = comp_df['metric'].unique()
    n_metrics = len(metrics_list)

    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics_list):
        ax = axes[i]
        metric_df = comp_df[comp_df['metric'] == metric].sort_values('avg_cliffs_delta', ascending=True)

        colours = ['#4CAF50' if sig else '#BDBDBD' for sig in metric_df['significant']]
        bars = ax.barh(metric_df['method'], metric_df['avg_cliffs_delta'],
                       color=colours, edgecolor='white', height=0.6)

        # add value labels
        for bar, val in zip(bars, metric_df['avg_cliffs_delta']):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=10)

        # effect size thresholds
        ax.axvline(x=0.147, color='orange', linestyle=':', alpha=0.5, label='Small (0.147)')
        ax.axvline(x=0.33, color='red', linestyle=':', alpha=0.5, label='Medium (0.33)')
        ax.axvline(x=0.474, color='darkred', linestyle=':', alpha=0.5, label='Large (0.474)')

        metric_label = metric.replace('_', ' ').title()
        ax.set_xlabel("Avg |Cliff's delta|")
        ax.set_title(f'{metric_label}')
        ax.legend(fontsize=8, loc='lower right')

    fig.suptitle("Model Comparison: Average Effect Sizes\n(green = significant KW test, grey = not significant)",
                 fontsize=14)
    plt.tight_layout()

    path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")
    return path





# CHART 7: Regime timeline with price overlay (section 4.3)
# HMM regime coloring behind the actual stock price


def plot_regime_with_price(feature_df: pd.DataFrame, price_df: pd.DataFrame,
                            method_name: str = "hmm", ticker: str = "",
                            output_dir: str = "output/charts"):
    # regime coloured background behind stock price line
    # shows whether regime changes align with price movements

    os.makedirs(output_dir, exist_ok=True)

    if 'cluster_label' not in feature_df.columns:
        print("  no cluster labels for regime-price overlay - skipping")
        return None

    fig, ax = plt.subplots(figsize=(16, 7))

    # get dates from feature matrix
    if 'trading_day' in feature_df.columns:
        dates = pd.to_datetime(feature_df['trading_day'])
    elif 'published_day' in feature_df.columns:
        dates = pd.to_datetime(feature_df['published_day'])
    else:
        print("  no date column - skipping regime price overlay")
        return None

    labels = feature_df['cluster_label'].values
    palette = sns.color_palette("deep", n_colors=10)
    noise_colour = 'lightgray'

    # draw regime background colours
    for i in range(len(dates)):
        lbl = int(labels[i])
        colour = noise_colour if lbl == -1 else palette[lbl % len(palette)]
        ax.axvspan(dates.iloc[i] - pd.Timedelta(hours=12),
                   dates.iloc[i] + pd.Timedelta(hours=12),
                   color=colour, alpha=0.25)

    # overlay the stock price
    if price_df is not None and not price_df.empty:
        if 'date' in price_df.columns:
            price_dates = pd.to_datetime(price_df['date'])
        else:
            price_dates = pd.to_datetime(price_df.index)

        if 'Close' in price_df.columns:
            close_col = 'Close'
        elif 'close' in price_df.columns:
            close_col = 'close'
        else:
            close_col = None

        if close_col:
            ax.plot(price_dates, price_df[close_col], color='black', linewidth=1.5,
                    alpha=0.8, label='Close Price')
            ax.set_ylabel('Close Price ($)')

    # add regime legend for understanding

    unique_labels = sorted(feature_df[feature_df['cluster_label'] >= 0]['cluster_label'].unique())
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=palette[l % len(palette)], alpha=0.4) for l in unique_labels]
    legend_labels = [f'Regime {l}' for l in unique_labels]
    if (feature_df['cluster_label'] == -1).any():
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=noise_colour, alpha=0.4))
        legend_labels.append('Noise')

    ax.legend(legend_handles, legend_labels, loc='upper left', fontsize=9)

    ax.set_xlabel('Date')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
    plt.xticks(rotation=45)

    title_suffix = f" ({method_name.upper()})" if method_name else ""
    ax.set_title(f'Regime Timeline with Price{title_suffix}{" - " + ticker if ticker else ""}')
    plt.tight_layout()

    fname = f'regime_price_overlay_{method_name}.png'
    path = os.path.join(output_dir, fname)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")
    return path


#  generate all chapter 4 charts at once


def generate_chapter4_charts(results_dict: Dict[str, pd.DataFrame],
                              validation_dict: Dict[str, Dict],
                              articles_df: Optional[pd.DataFrame] = None,
                              price_df: Optional[pd.DataFrame] = None,
                              hmm_transition_df: Optional[pd.DataFrame] = None,
                              ticker: str = "",
                              output_dir: str = "output/chapter4_charts"):
    # generates all 7 chapter 4 charts
    #
    # results_dict: {'kmeans': clustered_df, 'hdbscan': df, 'gmm': df, 'hmm': df}
    # validation_dict: {'kmeans_vader': validate_output, 'hmm_vader': ..., etc}
    # articles_df: raw articles with both vader_compound and finbert_compound
    # price_df: price data for the overlay chart
    # hmm_transition_df: output of hmm_clusterer.get_transition_matrix()

    print(f"\n{'=' * 50}")
    print("GENERATING CHAPTER 4 VISUALISATIONS")
    print(f"{'=' * 50}")

    os.makedirs(output_dir, exist_ok=True)
    generated = []

    # chart 1: scorer comparison scatter
    if articles_df is not None:
        path = plot_scorer_comparison(articles_df, ticker, output_dir)
        if path:
            generated.append(path)

    # chart 2: scorer distribution comparison
    if articles_df is not None:
        path = plot_scorer_distributions(articles_df, ticker, output_dir)
        if path:
            generated.append(path)

    # chart 3: regime timelines stacked
    if results_dict:
        path = plot_regime_timelines(results_dict, ticker, output_dir)
        if path:
            generated.append(path)

    # chart 4: return distributions for each method
    for method_name, method_df in results_dict.items():
        path = plot_return_distributions(method_df, method_name, ticker, output_dir)
        if path:
            generated.append(path)

    # chart 5: HMM transition matrix
    if hmm_transition_df is not None:
        path = plot_transition_matrix(hmm_transition_df, ticker, output_dir)
        if path:
            generated.append(path)

    # chart 6: model comparison
    if validation_dict:
        path = plot_model_comparison(validation_dict, output_dir)
        if path:
            generated.append(path)

    # chart 7: regime with price overlay (use HMM if available, else first method)
    if results_dict:
        overlay_method = 'hmm' if 'hmm' in results_dict else list(results_dict.keys())[0]
        path = plot_regime_with_price(results_dict[overlay_method], price_df,
                                      overlay_method, ticker, output_dir)
        if path:
            generated.append(path)

    print(f"\nGenerated {len(generated)} chapter 4 charts in {output_dir}/")
    return generated


# quick test with fake data
if __name__ == "__main__":

    np.random.seed(42)
    n = 60

    # fake feature matrix
    dates = pd.bdate_range('2026-01-05', periods=n)
    fake_df = pd.DataFrame({
        'trading_day': dates,
        'published_day': dates,
        'vader_mean': np.random.normal(0.1, 0.2, n),
        'daily_return': np.random.normal(0.001, 0.01, n),
        'realised_volatility_5d': np.random.uniform(0.005, 0.02, n),
        'cluster_label': np.random.choice([0, 1, 2], n),
    })

    # fake articles with both scorers
    fake_articles = pd.DataFrame({
        'vader_compound': np.random.normal(0.1, 0.4, 200),
        'finbert_compound': np.random.normal(0.05, 0.5, 200),
    })

    # fake price data
    fake_prices = pd.DataFrame({
        'date': dates,
        'Close': 100 + np.cumsum(np.random.randn(n) * 0.5),
    })

    # fake transition matrix
    fake_trans = pd.DataFrame(
        [[0.9, 0.05, 0.05], [0.04, 0.88, 0.08], [0.06, 0.06, 0.88]],
        index=['State 0', 'State 1', 'State 2'],
        columns=['State 0', 'State 1', 'State 2'],
    )

    # fake validation results
    from analysis.statistical_validation import validate_clusters
    fake_val = validate_clusters(fake_df)

    results = {'kmeans': fake_df.copy(), 'hdbscan': fake_df.copy(),
               'gmm': fake_df.copy(), 'hmm': fake_df.copy()}
    validations = {'kmeans_vader': fake_val, 'hdbscan_vader': fake_val,
                   'gmm_vader': fake_val, 'hmm_vader': fake_val}

    generate_chapter4_charts(
        results_dict=results,
        validation_dict=validations,
        articles_df=fake_articles,
        price_df=fake_prices,
        hmm_transition_df=fake_trans,
        ticker="TEST",
        output_dir="data/chapter4_charts",
    )
