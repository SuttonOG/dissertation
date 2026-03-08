

# Module for VADER Sentiment scoring, with a range from most negative_articles to positive_articles ( -1, 0 , +1 positive_articles, neutral_articles, negative_articles)
# Scores content (if scraping successful i.e not None) and Title, gives a score


import pandas as pd
from typing import Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderScorer:
    # Sentiment Scoring module

    def __init__(self):
        self.analyser = SentimentIntensityAnalyzer()

    # Scores single piece of text
    def score_text(self, text: str) -> dict:

        
        if not text or not isinstance(text, str):
            return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}        # if text is empty, return dict with 0 for all

        return self.analyser.polarity_scores(text)                              # else, return polarity_score of text using sentiment analyser

    # Scores all articles in a dataframe
    def score_dataframe(self, df: pd.DataFrame,
                        use_content: bool = True) -> pd.DataFrame:
        
        #Scoring strat = if use_content = True + content exists, score content from df, otherwise fall back to title. Score title sep


        # ARGS: df - contains columns 'title' and 'content' (check if none). use_content = whether to use article's body if scraping worked

        # Returns
        # vader_title_compound - the compound score from title
        # vader_content_compound - the compound score of content or NaN if scraping failed
        # vader_compound - primary score overall, content if it was available or title 
        # vader_ source - shows where the score came from

        df = df.copy()

        print(f"Calculating VADER Sentiment score for {len(df)} articles")

        # score all titles
        title_scores = df['title'].apply(self.score_text)                           # score title first
        df['vader_title_compound'] = title_scores.apply(lambda x: x['compound'])    

        # score content where available
        if use_content and 'content' in df.columns:
            content_scores = df['content'].apply(
                lambda x: self.score_text(x) if pd.notna(x) else None       # score title if not empty, else return None
            )
            df['vader_content_compound'] = content_scores.apply(            # score scraped content if exists else None
                lambda x: x['compound'] if x else None
            )
        else:
            df['vader_content_compound'] = None

        # build primary score: prefer content, fall back to title if content wasnt extracted
        def pick_primary(row):
            if use_content and pd.notna(row.get('vader_content_compound')):
                return row['vader_content_compound'], 'content'
            return row['vader_title_compound'], 'title'

        primary = df.apply(pick_primary, axis=1, result_type='expand')
        primary.columns = ['vader_compound', 'vader_source']
        df['vader_compound'] = primary['vader_compound']
        df['vader_source'] = primary['vader_source']

        # add component scores for primary text
        def get_components(row):
            if row['vader_source'] == 'content' and pd.notna(row.get('content')):
                scores = self.score_text(row['content'])
            else:
                scores = self.score_text(row['title'])
            return scores['neg'], scores['neu'], scores['pos']

        components = df.apply(get_components, axis=1, result_type='expand')     # add list of scores as columns via 'expand'
        components.columns = ['vader_neg', 'vader_neu', 'vader_pos']
        df = pd.concat([df, components], axis=1)

        # print summary
        content_scored = (df['vader_source'] == 'content').sum()
        title_scored = (df['vader_source'] == 'title').sum()
        mean_score = df['vader_compound'].mean()

        print(f"  Scored from content: {content_scored}")
        print(f"  Scored from title:   {title_scored}")
        print(f"  Mean compound score: {mean_score:.4f}")
        print(f"  Score range: [{df['vader_compound'].min():.4f}, {df['vader_compound'].max():.4f}]")

        # sentiment distribution
        positive_articles = (df['vader_compound'] > 0.05).sum()
        negative_articles = (df['vader_compound'] < -0.05).sum()
        neutral_articles = len(df) - positive_articles - negative_articles
        print(f"  Distribution: {positive_articles} positive_articles, {neutral_articles} neutral_articles, {negative_articles} negative_articles")

        return df


if __name__ == "__main__":
    # test with sample texts
    scorer = VaderScorer()

    # Test with different Nvidia texts to see if sentiment scoring working successfully.
    test_texts = [
        "NVIDIA stock surges after record earnings beat expectations",
        "Markets crash as trade war fears intensify amid tariff threats",
        "Federal Reserve holds interest rates steady at current levels",
        "Company announces massive layoffs amid restructuring concerns",
        "Tech giant reports strong revenue growth driven by AI demand",
    ]

    print("VADER Sentiment Test")
    print("-" * 60)
    for text in test_texts:
        scores = scorer.score_text(text)
        print(f"  [{scores['compound']:+.4f}] {text}")

    # test with a DataFrame if a CSV exists
    import os
    test_csv = "data/articles_NVDA_2d.csv"
    
    
    if os.path.exists(test_csv):
        print(f"\n{'=' * 60}")
        print(f"Scoring articles from {test_csv}")
        print(f"{'=' * 60}")
        df = pd.read_csv(test_csv)
        scored_df = scorer.score_dataframe(df)

        print(f"\nSample scored articles:")
        print("-" * 60)
        for _, row in scored_df.head(10).iterrows():
            print(f"  [{row['vader_compound']:+.4f}] ({row['vader_source']}) {row['title'][:70]}")