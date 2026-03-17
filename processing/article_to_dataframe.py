import pandas as pd
from dataclasses import dataclass, field, asdict
# from data_collection.models import NewsArticle
from data_collection.models import NewsArticle
# used to convert a list of news article objects to a dataframe
from typing import List


def convert_articles_to_dataframe(articles : List[NewsArticle]) -> pd.DataFrame:

    if not articles:
        print(f"No articles were found to convert")
        return pd.DataFrame()       # return empty instance 


    
    # added debugging for if articles is None

    if not articles:
        print("No articles were found for converting")
        return None
    
    # convert to list of dictionarys, from list of newsarticle objects
    dict_articles = [asdict(article) for article in articles]

    # NOW we have list of dictionary, convert to dataframe

    df = pd.DataFrame.from_records(dict_articles)

    if "published_date" in df.columns:
        df["published_date"] = pd.to_datetime(df["published_date"],utc=True, errors="coerce")

        # helper col for aggregation later
        df["published_day"] = df["published_date"].dt.date      # remove the gdelt hours minutes seconds after YMD

                                   # just to track that these came from GDELT, may be useful in the future
        df['origin'] = df['source'].apply(lambda s: 'GDELT' if str(s).startswith('gdelt_') else 'RSS')

        df["source"] = df["source"].str.replace("^gdelt_","",regex=True)    # remove gdelt_ from source to see the exact source


        # remove some duplicates or else its biased

        df["title_norm"] = df["title"].str.lower().str.strip()
        
        assert df["published_day"].notna().all()                # check to ensure logic works
        
        #create df with duplicates removed beforehand
        duplicates_removed_df = df.drop_duplicates(subset=["title_norm","published_day"]).copy()

        # calculate duplicate count 
        duplicate_count = len(df) - len(duplicates_removed_df)

        # display to user number of duplicates removed
        print(f"Removed {duplicate_count} duplicate articles (same title, same date of publishing)")

        # additioanl cleaning, to prevent the csv being broken by the new lines etc
        if 'content' in duplicates_removed_df.columns:
            duplicates_removed_df['content'] = duplicates_removed_df['content'].str.replace('\n', ' ', regex=False)
            duplicates_removed_df['content'] = duplicates_removed_df['content'].str.replace('\r', ' ', regex=False)
            duplicates_removed_df['content'] = duplicates_removed_df['content'].str.replace('  +', ' ', regex=True)

        # remove paywall/subscription content which is sometimes extracted  - not usable for analysis
        if 'content' in duplicates_removed_df.columns:
            paywall_phrases = [
                'subscribe to unlock',
                'try unlimited access',
                'complete digital access',
                'explore our full range of subscriptions',
                'pay a year upfront',
                'already have access via your university',
                'sign in to read this article',
                'this content is for subscribers',
                'continue reading for free',
                'some offers on this page are from advertisers',
                'find out how much you could earn',
            ]   # to add to these as more extractions occur to ensure we arent adding sentiment analysis to paywall text

            def check_for_paywall(text):
                if pd.isna(text) or not text:       # skip if content empty
                    return False
                text_lower = text.lower()
                return any(phrase in text_lower for phrase in paywall_phrases)      # if a phrase exists, paywall present, remove

            paywall_count = duplicates_removed_df['content'].apply(check_for_paywall).sum()
            if paywall_count > 0:
                duplicates_removed_df.loc[duplicates_removed_df['content'].apply(check_for_paywall), 'content'] = None
                print(f"Removed {paywall_count} paywall articles (content will be set to None, will use title only)")

        return duplicates_removed_df                            # return deduped dataframe

