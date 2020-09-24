import requests 
from typing import List
from bs4 import BeautifulSoup
from app.api.preprocessing_and_sentiment import get_sentiment_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def extract_hn_page_urls(
        hn_seed: str = 'https://news.ycombinator.com/newcomments?',
        page_limit: int = 5) -> List[str]:
    """Extract Hacker News comment page URLs, which are based
    on the total number of Hacker News comments."""

    urls = []
    site = hn_seed
    while len(urls) < page_limit:
        page = requests.get(site)
        soup = BeautifulSoup(page.content, features='html.parser')
        page_url = soup.find("a", {"class": "morelink"})['href'].split('?')[1]
        urls.append(hn_seed + page_url)
        site = hn_seed + page_url

    return urls


def get_hn_users_comments_scores(urls: str):
    total_com_scores = []
    for url in urls:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, features='html.parser')
        user_coms = zip(soup.find_all('a', class_='hnuser'),
                        soup.find_all('div', class_='comment'))

        for user, com in user_coms:
            score = get_sentiment_score(com.get_text())
            total_com_scores.append([user.get_text(), score])

    total_com_scores = total_com_scores = sorted(
        total_com_scores, key=lambda x: x[1])

    response_dict = {k + 1: [v[0], v[1]]
                     for k, v in enumerate(total_com_scores[:100])}
    return response_dict
