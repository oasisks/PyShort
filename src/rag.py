import aiohttp
import asyncio
import os
import random
import re
import requests
import ssl
import yfinance as yf

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import Dict, List, Tuple

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
        print(ticker_obj.get_news())
        return ticker_obj.get_news()
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []


async def fetch_news_content(session: aiohttp.ClientSession, url: str, title: str) -> Tuple[str, str]:
    """
    Performs the request on the URL and grab the content of the yahoo finance news source
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


async def process_tickers(tickers: List[str]) -> Dict[str, Dict[str, str]]:
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
        all_ticker_news = {result[0]: result[1] for result in results}

        return all_ticker_news


def get_ticker_news(tickers: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Given a ticker symbol, it will return a dict of headlines with its respective content
    :param tickers: a list of ticker symbol
    :return: dict
    """

    results = asyncio.run(process_tickers(tickers))
    return results


if __name__ == '__main__':
    tickers = ["GOOG"]

    alL_news = get_ticker_news(tickers)
    for ticker, news in alL_news.items():
        for title, content in news.items():
            print(f"Title: {title}")

            print(f"Content: {content}")
