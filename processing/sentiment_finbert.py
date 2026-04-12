
# FinBERT Sentiment Analysis for Financial text

"Simple implementation via hugginface pipeline"
"Matches Same interface as VaderScorer so can be easily swapped"
"Same usecase as Vader, create scorer, then score dataframe"


import pandas as pd
from transformers import pipeline


class FinBertScorer:
    """
    FinBERT sentiment scorer using HuggingFace pipeline.
    Same interface as VaderScorer.
    """
    
    def __init__(self):
        """Load the FinBERT model (downloads ~420MB first time)."""
        print("Loading FinBERT model...")
        self.classifier = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512
        )
        print("FinBERT model loaded!")
    
    def score_text(self, text: str) -> dict:

        # score single piece of text - same as vader
        # return dict with neg, neu pos, compound keys (same as vader)



        # Handle empty text (same as VADER)
        if not text or not isinstance(text, str):
            return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
        
        # Get prediction from pipeline
        result = self.classifier(text)[0]
        label = result['label']
        score = result['score']
        
        # Convert to VADER-style output
        if label == 'positive':
            return {'neg': 0.0, 'neu': 1 - score, 'pos': score, 'compound': score}
        elif label == 'negative':
            return {'neg': score, 'neu': 1 - score, 'pos': 0.0, 'compound': -score}
        else:  # neutral
            return {'neg': 0.0, 'neu': score, 'pos': 0.0, 'compound': 0.0}
    
    def score_dataframe(self, df: pd.DataFrame, use_content: bool = True) -> pd.DataFrame:
        # score all articles in df, same interface as vaderscore

        # pass df and use_content (whether use body if we have)
        # return df with sentiment cols added

        df = df.copy()
        
        print(f"Calculating FinBERT sentiment for {len(df)} articles")
        
        # Score all titles (same pattern as VADER)
        title_scores = df['title'].apply(self.score_text)
        df['finbert_title_compound'] = title_scores.apply(lambda x: x['compound'])
        
        # Score content where available (same pattern as VADER)
        if use_content and 'content' in df.columns:
            content_scores = df['content'].apply(
                lambda x: self.score_text(x) if pd.notna(x) else None
            )
            df['finbert_content_compound'] = content_scores.apply(
                lambda x: x['compound'] if x else None
            )
        else:
            df['finbert_content_compound'] = None
        
        # Build primary score: prefer content, fall back to title (same as VADER)
        def pick_primary(row):
            if use_content and pd.notna(row.get('finbert_content_compound')):
                return row['finbert_content_compound'], 'content'
            return row['finbert_title_compound'], 'title'
        
        primary = df.apply(pick_primary, axis=1, result_type='expand')
        primary.columns = ['finbert_compound', 'finbert_source']
        df['finbert_compound'] = primary['finbert_compound']
        df['finbert_source'] = primary['finbert_source']
        
        # Add component scores for primary text (same pattern as VADER)
        def get_components(row):
            if row['finbert_source'] == 'content' and pd.notna(row.get('content')):
                scores = self.score_text(row['content'])
            else:
                scores = self.score_text(row['title'])
            return scores['neg'], scores['neu'], scores['pos']
        
        components = df.apply(get_components, axis=1, result_type='expand')
        components.columns = ['finbert_neg', 'finbert_neu', 'finbert_pos']
        df = pd.concat([df, components], axis=1)
        
        # Print summary (same as VADER)
        content_scored = (df['finbert_source'] == 'content').sum()
        title_scored = (df['finbert_source'] == 'title').sum()
        mean_score = df['finbert_compound'].mean()
        
        print(f"  Scored from content: {content_scored}")
        print(f"  Scored from title:   {title_scored}")
        print(f"  Mean compound score: {mean_score:.4f}")
        print(f"  Score range: [{df['finbert_compound'].min():.4f}, {df['finbert_compound'].max():.4f}]")
        
        # Sentiment distribution
        positive_articles = (df['finbert_compound'] > 0.05).sum()
        negative_articles = (df['finbert_compound'] < -0.05).sum()
        neutral_articles = len(df) - positive_articles - negative_articles
        print(f"  Distribution: {positive_articles} positive, {neutral_articles} neutral, {negative_articles} negative")
        
        return df


if __name__ == "__main__":
    # Test with sample texts (same as VADER's test)
    scorer = FinBertScorer()
    
    test_texts = [
        "NVIDIA stock surges after record earnings beat expectations",
        "Markets crash as trade war fears intensify amid tariff threats",
        "Federal Reserve holds interest rates steady at current levels",
        "Company announces massive layoffs amid restructuring concerns",
        "Tech giant reports strong revenue growth driven by AI demand",
    ]
    
    print("\nFinBERT Sentiment Test")
    print("-" * 60)
    for text in test_texts:
        scores = scorer.score_text(text)
        print(f"  [{scores['compound']:+.4f}] {text}")
    
    # Test with DataFrame if CSV exists
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
            print(f"  [{row['finbert_compound']:+.4f}] ({row['finbert_source']}) {row['title'][:70]}")``