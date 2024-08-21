import datetime

from devtools import pprint
from typing import List, Dict
import yfinance as yf


def get_float(stocks: List[str], days: List[datetime.datetime]) -> Dict[datetime.datetime, Dict[str, int]]:
    """
    Returns the float values by day for each stock

    Stocks must be valid stocks. Will result in an error if the name of the stock does not exist
    :param stocks: a list of stock symbols
    :param days: A list of days
    :return: The float values for each stock for each day
    """

    for _ in stocks:
        stock = yf.Ticker(_)
        pprint(stock.info)


if __name__ == '__main__':
    stocks = ["MSFT"]
    days = [datetime.datetime.now()]

    get_float(stocks, days)
