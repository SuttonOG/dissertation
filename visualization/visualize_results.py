# visualise_results.py - generates all the charts for the demo + dissertation
# uses matplotlib and seaborn to make the plots look decent
# saves everything to an output folder so can use them in the report later

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Optional
import os


# make the plots look a bit nicer than the default matplotlib style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


def plot_sentiment_over_time(df: pd.DataFrame, ticker: str = "",
                              output_dir: str = "output/charts"):
    
    # plot daily avg sentiment over time
    # show general trend of positive / mehgative news over time
    # 


    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    dates = pd.to_datetime(df['published_day'])
    sentiment = df['vader_mean']



    # plot the line
    ax.plot(dates, sentiment, color='#2196F3', linewidth=1.5, marker='o',
            markersize=4, label='Daily mean sentiment', zorder=3)



    # fill above/below zero to make it visually obvious whats positive vs negative
    ax.fill_between(dates, sentiment, 0,
                    where=(sentiment >= 0), color='#4CAF50', alpha=0.15, label='Positive')
    ax.fill_between(dates, sentiment, 0,
                    where=(sentiment < 0), color='#F44336', alpha=0.15, label='Negative')


    # zero line for reference
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Date')
    ax.set_ylabel('Mean VADER Compound Score')
    ax.set_title(f'Daily News Sentiment Over Time{" - " + ticker if ticker else ""}')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = os.path.join(output_dir, 'sentiment_over_time.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")
    return path


def plot_sentiment_vs_returns(df: pd.DataFrame, ticker: str = "",
                               output_dir: str = "output/charts"):
    
    # scatter plot of senrtiment vs next-day stock returns
    # key relationship for investigation, if correlation -> main finding

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    x = df['vader_mean']
    y = df['daily_return']

    # colour points by whether return was positive or negative
    colours = ['#4CAF50' if r >= 0 else '#F44336' for r in y]
    ax.scatter(x, y, c=colours, alpha=0.6, edgecolors='white', linewidth=0.5, s=60)



    # add a trend line to see if theres any relationship
    # using numpy polyfit for a simple linear fit
    if len(x.dropna()) > 2:
        mask = x.notna() & y.notna()
        z = np.polyfit(x[mask], y[mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(x_line, p(x_line), '--', color='#FF9800', linewidth=2,
                label=f'Trend (slope={z[0]:.4f})')

        # also calculate the correlation - important for the writeup
        corr = x[mask].corr(y[mask])
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.8)



    ax.set_xlabel('Mean Daily Sentiment (VADER)')
    ax.set_ylabel('Daily Stock Return')
    ax.set_title(f'Sentiment vs Daily Returns{" - " + ticker if ticker else ""}')
    ax.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, 'sentiment_vs_returns.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")
    return path




def plot_cluster_scatter(df: pd.DataFrame, ticker: str = "",
                          output_dir: str = "output/charts"):
    
    # scatter plot to show clusters - colour coded by cluster label 
    # x axis = sentiment, y azis = returns
    # main visual to show clustering findings 
    # noise points = cluster -1 (grey)
   

    os.makedirs(output_dir, exist_ok=True)
    

    if 'cluster_label' not in df.columns:
        print("  skipping cluster scatter - no cluster_label column found")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # separate noise from actual clusters
    noise = df[df['cluster_label'] == -1]
    clustered = df[df['cluster_label'] >= 0]

    # plot noise points first (behind everything)
    if not noise.empty:
        ax.scatter(noise['vader_mean'], noise['daily_return'],
                   c='lightgray', alpha=0.4, s=30, label='Noise', zorder=1)

    # plot each cluster with a different colour
    # using a nice colour palette that works for up to ~10 clusters
    palette = sns.color_palette("deep", n_colors=max(clustered['cluster_label'].nunique(), 1))

    for i, label in enumerate(sorted(clustered['cluster_label'].unique())):
        cluster = clustered[clustered['cluster_label'] == label]
        ax.scatter(cluster['vader_mean'], cluster['daily_return'],
                   c=[palette[i % len(palette)]], s=70, alpha=0.7,
                   edgecolors='white', linewidth=0.5,
                   label=f'Cluster {label} (n={len(cluster)})', zorder=2)

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.8)

    ax.set_xlabel('Mean Daily Sentiment (VADER)')
    ax.set_ylabel('Daily Stock Return')
    ax.set_title(f'Sentiment Clusters{" - " + ticker if ticker else ""}')
    ax.legend(loc='best')
    plt.tight_layout()

    path = os.path.join(output_dir, 'cluster_scatter.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")
    return path



def plot_cluster_profiles(df: pd.DataFrame, ticker: str = "",
                           output_dir: str = "output/charts"):
    
    # bar chart comparing key metrics across clusters 
    # makes easy to see what a clusters characteristics are e.g cluster 0 = positive sentiment + low vow, cluster 1 = negative + high vol

    os.makedirs(output_dir, exist_ok=True)

    if 'cluster_label' not in df.columns:
        print("  skipping cluster profiles - no cluster_label column found")
        return None


    # only use actual clusters, not noise
    clustered = df[df['cluster_label'] >= 0]
    if clustered.empty:
        print("  no clusters to profile")
        return None

    # metrics to compare across clusters
    metrics = ['vader_mean', 'daily_return', 'realised_volatility_5d', 'positive_ratio']
    available = [m for m in metrics if m in clustered.columns]

    if not available:
        print("  no metrics available for profiling")
        return None

    # nicer labels for the chart
    labels = {
        'vader_mean': 'Avg Sentiment',
        'daily_return': 'Avg Return',
        'realised_volatility_5d': 'Avg Volatility (5d)',
        'positive_ratio': 'Positive Article Ratio',
    }

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 6))
    if len(available) == 1:
        axes = [axes]       # matplotlib is weird with single subplot

    palette = sns.color_palette("deep", n_colors=clustered['cluster_label'].nunique())

    for i, metric in enumerate(available):
        means = clustered.groupby('cluster_label')[metric].mean()
        bars = axes[i].bar(means.index.astype(str), means.values,
                           color=palette[:len(means)], edgecolor='white')
        axes[i].set_title(labels.get(metric, metric))
        axes[i].set_xlabel('Cluster')

        # add value labels on top of bars cos its easier to read
        for bar, val in zip(bars, means.values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle(f'Cluster Profiles{" - " + ticker if ticker else ""}', fontsize=15)
    plt.tight_layout()

    path = os.path.join(output_dir, 'cluster_profiles.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")
    return path



def plot_sentiment_distribution(articles_df: pd.DataFrame, ticker: str = "",
                                 output_dir: str = "output/charts"):
    
    # histogram of individual article sentiment scores
    # shows overall distribution -> most tends to be positive for financial news
    # good for methodol sect

    os.makedirs(output_dir, exist_ok=True)

    if 'vader_compound' not in articles_df.columns:
        print("  skipping sentiment distribution - no vader_compound column")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    scores = articles_df['vader_compound'].dropna()

    ax.hist(scores, bins=40, color='#2196F3', alpha=0.7, edgecolor='white')
    ax.axvline(x=scores.mean(), color='#F44336', linestyle='--', linewidth=2,
               label=f'Mean = {scores.mean():.3f}')
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)

    ax.set_xlabel('VADER Compound Score')
    ax.set_ylabel('Number of Articles')
    ax.set_title(f'Distribution of Article Sentiment Scores{" - " + ticker if ticker else ""}')
    ax.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, 'sentiment_distribution.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved: {path}")
    return path


def generate_all_charts(feature_matrix: pd.DataFrame,
                        articles_df: Optional[pd.DataFrame] = None,
                        ticker: str = "",
                        output_dir: str = "output/charts"):
    
    # func to generate all charts at once 


    print(f"\n{'=' * 50}")
    print("GENERATING VISUALISATIONS")
    print(f"{'=' * 50}")

    os.makedirs(output_dir, exist_ok=True)
    generated = []

    # 1. sentiment over time (needs the aggregated daily data)
    if 'published_day' in feature_matrix.columns and 'vader_mean' in feature_matrix.columns:
        path = plot_sentiment_over_time(feature_matrix, ticker, output_dir)
        if path:
            generated.append(path)

    # 2. sentiment vs returns ( key relationship)
    if 'vader_mean' in feature_matrix.columns and 'daily_return' in feature_matrix.columns:
        path = plot_sentiment_vs_returns(feature_matrix, ticker, output_dir)
        if path:
            generated.append(path)

    # 3. cluster scatter (if clustering has been done)
    if 'cluster_label' in feature_matrix.columns:
        path = plot_cluster_scatter(feature_matrix, ticker, output_dir)
        if path:
            generated.append(path)

        # 4. cluster profiles
        path = plot_cluster_profiles(feature_matrix, ticker, output_dir)
        if path:
            generated.append(path)

    # 5. article-level sentiment distribution (if we have individual articles)
    if articles_df is not None and 'vader_compound' in articles_df.columns:
        path = plot_sentiment_distribution(articles_df, ticker, output_dir)
        if path:
            generated.append(path)

    print(f"\nGenerated {len(generated)} charts in {output_dir}/")
    return generated






# quick test using existing data
if __name__ == "__main__":
    # try loading from previous pipeline runs
    feature_path = "data/feature_matrix.csv"
    articles_path = "data/articles_NVDA_2d.csv"

    if os.path.exists(feature_path):
        fm = pd.read_csv(feature_path)
        articles = pd.read_csv(articles_path) if os.path.exists(articles_path) else None

        generate_all_charts(fm, articles, ticker="NVDA")
    else:
        print("no feature matrix found - run the pipeline first")