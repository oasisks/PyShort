import os
import pandas as pd
import requests

import requests
import yfinance as yf
from datetime import datetime
from devtools import pprint
from typing import List, Dict
from finviz.main_func import get_stock


def get_recent_time_series(stocks: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Returns the float values by day for each stock

    Stocks must be valid stocks. Will result in an error if the name of the stock does not exist
    :param stocks: a list of stock symbols

    :return: The float values for each stock for each day
    """
    # For now, we will just see if it works for getting the float for today only
    tickers = yf.Tickers(" ".join(stocks))

    return {ticker: ticker_obj.history(period="1mo") for ticker, ticker_obj in tickers.tickers.items()}


if __name__ == '__main__':
    stocks = ["MSFT", "GOOG"]
    get_recent_time_series(stocks)
