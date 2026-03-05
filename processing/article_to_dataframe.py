import pandas as pd
from dataclasses import dataclass, field, asdict
# from data_collection.models import NewsArticle
from data_collection.models import NewsArticle
# used to convert a list of news article objects to a dataframe


def convert_articles_to_dataframe(articles : NewsArticle) -> pd.DataFrame:

    if not articles:
        print(f"No articles were found to convert")
        return pd.DataFrame


    
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

        df['origin'] = "GDELT"                                # just to track that these came from GDELT, may be useful in the future

        df["source"] = df["source"].str.replace("^gdelt_","",regex=True)    # remove gdelt_ from source to see the exact source


        # remove some duplicates or else its biased

        df["title_norm"] = df["title"].str.lower().str.strip()
        
        assert df["published_day"].notna().all()                # check to ensure logic works
        
        #create df with duplicates removed beforehand
        duplicates_removed_df = df.drop_duplicates(subset=["title_norm","published_day"]) 

        # calculate duplicate count 
        duplicate_count = len(df) - len(duplicates_removed_df)

        # display to user number of duplicates removed
        print(f"Removed {duplicate_count} duplicate articles (same title, same date of publishing)")
      

        return duplicates_removed_df                            # return deduped dataframe

