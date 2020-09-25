import re
import requests 
from typing import List
from bs4 import BeautifulSoup
from app.api.preprocessing_and_sentiment import get_sentiment_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def get_hn_users_comments_scores(num: int = 100) -> List[str]:
    """Scrapes Hacker News for usernames and comments, 30 per page, and
    estimates users' sentiment scores. Default number to pull is 100."""

    total_users_comments_scores = []
    hn_seed = 'https://news.ycombinator.com/newcomments?'
    site = hn_seed
    while len(total_users_comments_scores) < num:
        res = requests.get(site, headers={'User-agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.content, features='html.parser')
        relative_path = soup.find('a', {'class': 'morelink'})[
            'href'].split('?')[1]

        users_and_comments = zip(soup.find_all('a', {'class': 'hnuser'}),
                                 soup.find_all('div', {'class': 'comment'}))

        for user, comment in users_and_comments:
            comment = comment.get_text().replace('\n', '').replace("\'", "")
            comment = comment.replace('/\\', '').replace('> ', '')
            # Remove URLs
            comment = re.sub(
                r'http[s]?(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', comment)
            score = get_sentiment_score(comment)
            total_users_comments_scores.append([user.get_text(),
                                                comment, 
                                                score])

        # Update site to be next page
        site = hn_seed + relative_path

    # Sort by scores in ascending order, most extreme negative first
    total_users_comments_scores = sorted(
        total_users_comments_scores, key=lambda x: x[2])

    response_dict = {k + 1: [v[0], v[1], v[2]]
                     for k, v in enumerate(total_users_comments_scores[:num])}

    return response_dict
