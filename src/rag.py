import aiohttp
import asyncio
import os
import random
import re
import ssl
import yfinance as yf
import stock_twits.stocktwits as stocktwits
import datetime
import json

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import Dict, List, Tuple
from model.RagModel import SummaryOutput, Article
from bert_score import score

from itertools import combinations

load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")

def calculate_bertscore(candidate, reference, model_path):
    """
    Calculate BERTScore using a custom fine-tuned BERT model.

    Args:
        candidate (str): The candidate summary.
        reference (str): The reference summary.
        model_path (str): Path to the fine-tuned BERT model.

    Returns:
        F1 scores.
    """
    # Calculate BERTScore
    P, R, F1 = score(
        [candidate],         # List of candidate summaries
        [reference],         # List of reference summaries
        model_type=model_path,  # Custom fine-tuned BERT model
        num_layers=12,        # Specify the layer if needed
        verbose=True
    )
    return F1.item()

async def fetch_news(ticker: str) -> List[Dict[str, str]]:
    """
    Fetches a list of news
    :param ticker: ticker symbol
    :return: a tuple containing the ticker symbol and also the list of news
    """
    ticker_obj = yf.Ticker(ticker)
    try:
        return ticker_obj.get_news()
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []


async def fetch_news_content(session: aiohttp.ClientSession, url: str, title: str) -> Tuple[str, str]:
    """
    Performs the request on the URL and grab the content of the yahoo finance news source
    :return:
    :param session: the current session
    :param url: article url
    :param title: article title
    :return:
    """
    try:
        async with session.get(url) as response:
            # await asyncio.sleep(random.uniform(1, 3))

            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")
            paragraphs = soup.find_all("p", class_="yf-1pe5jgt")  # The ID for article content

            return title, "\n".join([p.text for p in paragraphs])

    except Exception as e:
        print(f"Error fetching article content from {url}: {e}")
        return title, "Fetch Error"


async def process_ticker_news(session: aiohttp.ClientSession, ticker: str) -> Tuple[str, Dict[str, str]]:
    """
    Process each ticker and grab its news concurrently
    :param session: the client session
    :param ticker: the ticker symbol
    :return: returns a tuple containing the ticker symbol and also the content of the news with its title
    """
    news = await fetch_news(ticker)
    if not news:
        return ticker, {}

    # we only match urls with the /news because /m is a redirect to another platform
    pattern = r"^https://finance\.yahoo\.com/news"
    tasks = [fetch_news_content(session, article["link"], article["title"]) for article in news if
             re.match(pattern, article["link"])]
    results = await asyncio.gather(*tasks)

    articles = {result[0]: result[1] for result in results}

    return ticker, articles


async def process_tickers(tickers: List[str]) -> Dict[str, List[Article]]:
    """
    Process each ticker concurrently using asyncio
    :param tickers: a list of ticker symbols
    :return:
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive"
    }

    ssl_context = ssl.create_default_context()
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    jar = aiohttp.CookieJar()

    async with aiohttp.ClientSession(headers=headers, connector=connector, cookie_jar=jar, max_line_size=8190 * 2,
                                     max_field_size=8190 * 2) as session:
        tasks = [process_ticker_news(session, ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        all_ticker_news = {
            result[0]: [Article(headline=headline, content=content) for headline, content in result[1].items()]
            for result in results}

        return all_ticker_news



def get_ticker_news(tickers: List[str]) -> Dict[str, List[Article]]:
    """
    Given a ticker symbol, it will return a dict of headlines with its respective content
    :param tickers: a list of ticker symbol
    :return: dict
    """
    results = asyncio.run(process_tickers(tickers))

    model_path = "./finetuned_bert/bert-mlm"
    scores = []
    for ticker, news in results.items():
        if len(news) <= 5:
            continue
        
        for candidate, reference in combinations(news, 2):
            score = calculate_bertscore(candidate.content, reference.content, model_path)
            scores.append((candidate, reference, score))
        
        # Sort the scores by BERTScore (ascending)
        sorted_scores = sorted(scores, key=lambda x: x[2])
        selected_articles = []
        # At least 5 unique articles are selected
        for candidate, reference, score in sorted_scores:
            if candidate not in selected_articles:
                selected_articles.append(candidate)
            
            if reference not in selected_articles:
                selected_articles.append(reference)
        
            # Stop if we have selected 5 unique articles
            if len(selected_articles) >= 5:
                break
        
        results[ticker] = selected_articles

    return results


def get_ticker_tweets(tickers: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """
    Given a list of tickers, it will return a dict of tweets with its respective content. Sample 10 random tweets from each stock and sort them by date (newest first).

    :param tickers: a list of ticker symbol
    :return: dict {ticker: [tweets], ...}
    tweet: {id, body, created_at}
    """
    # grab the last 24 hours of tweets
    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=1)
    end_date = end_date.isoformat()
    start_date = start_date.isoformat()

    symbols_to_grab = {t: {"start_date": start_date, "end_date": end_date} for t in tickers}

    stock_messages = asyncio.run(stocktwits.fetch_all_stocks(symbols_to_grab))

    # randomly sample 10 tweets from each stock
    num_items_to_select = 10
    for stock, tweets in stock_messages.items():
        # get 10 only if have more than 10
        selected_tweets = random.sample(tweets, min(num_items_to_select, len(tweets)))
        stock_messages[stock] = sorted(selected_tweets, key=lambda x: x["created_at"], reverse=True)

    return stock_messages


def get_summary(tickers: List[str]) -> List[SummaryOutput]:
    """
    For each ticker within tickers, grab news article related to each ticker from yfinance and grab related tweets from
    stocktwits. Then pass each information source into the summary LLM to create a summary.
    :param tickers: list of ticker symbols (i.e. "GOOG")
    :return: A dictionary of the ticker symbols to a tuple of
    """
    all_tweets = get_ticker_tweets(tickers)
    all_news = get_ticker_news(tickers)

    print(json.dumps(all_tweets, indent=4))
    print(json.dumps(all_news, indent=4))


if __name__ == '__main__':
    tickers = ["AAPL", "AMZN"]

    # all_tweets = get_ticker_tweets(tickers)
    # for ticker, tweets in all_tweets.items():
    #     print(f"\nMessages for {ticker}:")
    #     for tweet in tweets:
    #         print(tweet["created_at"])
    #         print(tweet["body"])
    #         print("\n")

    articles = get_ticker_news(tickers)
    for ticker, news in articles.items():
        print(f"\nArticles for {ticker}:")
        print(len(news))
        for article in news:
            print(article.headline)
            print(article.content)
            print("\n")

    # get_summary(tickers)
