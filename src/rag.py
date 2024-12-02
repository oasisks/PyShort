import time

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
from llm.summary import create_file, create_batch_summarize, retrieve_batch_summarize, SummaryStatus, list_batches
from model.RagModel import SummaryOutput, Article

load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")


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
    Given a ticker symbol, it will return a dict of ticker with its respective news articles
    :param tickers: a list of ticker symbol
    :return: dict
    """
    results = asyncio.run(process_tickers(tickers))
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


def generate_news_summary_prompt(ticker: str) -> str:
    """
    A helper function to generate a news summary prompt for each ticker
    :param ticker: symbol
    :return: Returns the summary prompt for news
    """
    news_summary_prompt = (f"Please summarize the following noisy but possible news data extracted from the "
                           f"YahooFinance News for {ticker} stock, and extract keywords of the news. The news text can "
                           f"be very noisy since no filtering has been done. Provide a separate summary for each "
                           f"article and extract keywords for all. Format the answer as: Summary: Article 1: …, …, "
                           f"Article N: …, Keywords: … You may put ‘N/A’ if the noisy text does not have relevant "
                           f"information to extract.\nNews:\n")

    return news_summary_prompt


def generate_tweets_summary_prompt(ticker: str) -> str:
    """
    A helper function to generate a tweet summary for each ticker
    :param ticker: symbol
    :return: Returns the summary prompt for the tweets
    """
    tweets_summary_prompt = ("Instruction: Please summarize the following noisy but possible tweet posts extracted "
                             f"from StockTwits for {ticker} stock, and extract keywords from the tweets. The tweets' "
                             "text can be very noisy due to it being user-generated. Provide a separate summary for "
                             "each tweet and extract keywords for all. Format the answer as: Summary: Tweet 1: …, …, "
                             "Tweet N: …, Keywords: … You may put ’N/A’ if the noisy text does not have relevant "
                             "information to extract.\nTweets:\n")
    return tweets_summary_prompt


def get_summary(tickers: List[str]) -> List[SummaryOutput]:
    """
    For each ticker within tickers, grab news article related to each ticker from yfinance and grab related tweets from
    stocktwits. Then pass each information source into the summary LLM to create a summary.

    :param tickers: list of ticker symbols (i.e. "GOOG")
    :return: A dictionary of the ticker symbols to a tuple of
    """
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    all_news = get_ticker_news(tickers)
    news_inputs = [
        (ticker + "_news", generate_news_summary_prompt(ticker) + "\n".join(
            [f"{i + 1}. {article.headline}\n{article.content}" for i, article in enumerate(articles)]))
        for ticker, articles in all_news.items()]

    news_file_name = f"{current_time}_news.jsonl"
    news_bytes = create_file(news_inputs, news_file_name, summarize_tweet=False, in_bytes=True)
    news_batch = create_batch_summarize(file_name=news_file_name, input_bytes=news_bytes)

    all_tweets = get_ticker_tweets(tickers)
    tweets_inputs = [
        (ticker + "_tweets", generate_tweets_summary_prompt(ticker) + "\n".join(
            [f"{i + 1}. {tweet['body']} " for i, tweet in enumerate(tweets)]))
        for ticker, tweets in all_tweets.items()]
    tweets_file_name = f"{current_time}_tweets.jsonl"
    tweets_bytes = create_file(tweets_inputs, tweets_file_name, summarize_tweet=False, in_bytes=True)
    tweets_batch = create_batch_summarize(file_name=tweets_file_name, input_bytes=tweets_bytes)

    # once the batch is created, we will wait for 10 seconds until we return that the batch is still in process
    print(news_batch.id)
    print(tweets_batch.id)
    # wait_time = 10
    # i = 0
    # while i < wait_time:
    #     news_status = retrieve_batch_summarize(news_batch.id)
    #     tweets_status = retrieve_batch_summarize(tweets_batch.id)
    #
    #     if news_status == SummaryStatus.COMPLETED and tweets_status == SummaryStatus.COMPLETED:
    #         print("Finished processing and written on disk")
    #         break
    #
    #     time.sleep(1)
    #     i += 1


if __name__ == '__main__':
    tickers = ["GOOG", "AAPL"]

    # all_tweets = get_ticker_tweets(tickers)
    # for ticker, tweets in all_tweets.items():
    #     print(f"\nMessages for {ticker}:")
    #     for tweet in tweets:
    #         print(tweet["created_at"])
    #         print(tweet["body"])
    #         print("\n")

    # get_summary(tickers)
    # file = open("news_output.jsonl", "rb")

    # print(file)
    print(list_batches())
    retrieved_results = retrieve_batch_summarize("batch_674e22c75f0881908a914c4e949f4e2a")
    #
    for line in retrieved_results[1]:
        print(line)
