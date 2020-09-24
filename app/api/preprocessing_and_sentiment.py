import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def load_data(seed: str = f'https://github.com/Salty-Hackers/'
              'data-engineering/blob/main/Data/hn_',
              num: int = 11, limit: int = 500000) -> pd.DataFrame:

    assert isinstance(num, int), 'Num must be integer.'
    assert 0 <= num <= 11, 'Num must be between 0 and 11.'

    urls = [seed + f'{i}.csv?raw=true' for i in range(num)]
    cols = ['author', 'time_ts', 'text']
    df = pd.concat(
        (pd.read_csv(url, usecols=cols) for url in urls)) \
        .reset_index().drop(columns='index')

    df = df.rename(columns={'author': 'user', 
                            'time_ts': 'date_time', 
                            'text': 'comment'}
                   )

    # Return most recent non-NaN rows, default 500,000
    return df.dropna().sort_values(by='date_time', ascending=False)[:limit]


def preprocess(df: pd.DataFrame, text_col: str = 'comment',
               date_col: str = 'date_time') -> pd.DataFrame:
    """
    Takes dataframe, column (default 'text'). Returns df with no HTML, emails
    URLs, hexadecimal, or multi/leading/trailing spaces in that column.
    Renames text_col and date_col, drops microseconds from date_col, and 
    converts it to datetime.
    """

    # Make shallow copy to preserve original
    df = df[:]

    # Remove HTML tags
    df[text_col] = df[text_col].apply(
        lambda comment: re.sub(r'<.*?>', ' ', comment))

    # Remove emails
    df[text_col] = df[text_col].apply(
        lambda comment: re.sub(r'[\w.]+@\w+\.[a-z]{3}', '', comment))

    # Remove URLs
    df[text_col] = df[text_col].apply(
        lambda comment: re.sub(
            r'http[s]?(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
            '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', comment))

    # Remove punctuation encoding
    df[text_col] = df[text_col].apply(
        lambda comment: re.sub(r'&.*?;', '', comment))

    # Remove multi-spaces and leading/trailing whitespace characters
    df[text_col] = df[text_col].apply(
        lambda comment: re.sub(r'\s{2,}', ' ', comment).strip())

    # Remove microseconds from date_col
    df[date_col] = df[date_col].apply(
        lambda date: date.split('+')[0])

    # Convert to date_time and rename, drop date_col
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
    df = df.rename({date_col: 'date_time'})

    return df


analyzer = SentimentIntensityAnalyzer()


def get_sentiment_score(comment: str) -> float:
    """
    Returns normalized composite score from [-1, 1], weighted by intensity.
    -1 is most negative, 1 is most positive.
    """
    sentiment_dict = analyzer.polarity_scores(comment)
    return sentiment_dict['compound']


def get_sentiment(score: float, thresh=0.05) -> str:
    """
    Engineers sentiment (positive, negative, neutral) 
    based on sentiment score [-1, 1].

    Default thresholds:
      positive: compound score >= 0.05
      neutral: 0.05 > compound score > -0.05
      negative: compound score <= -0.05
    """
    assert isinstance(
        score, (float, int)), 'Sentiment score must be float or int.'
    assert isinstance(
        thresh, (float, int)), 'Threshold must be float or int.'

    assert -1 <= score <= 1, 'Sentiment score must be between -1 and 1.'
    assert 0 < thresh < 1, 'Threshold must be between 0 and 1.'

    if score >= thresh:
        return 'positive'
    elif score <= -thresh:
        return 'negative'

    return 'neutral'
