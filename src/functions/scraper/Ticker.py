import datetime
import os
import requests

import requests
from devtools import pprint
from dotenv import load_dotenv
from typing import List, Dict
from finviz.main_func import get_stock

load_dotenv()


def get_float(stocks: List[str], days: List[datetime.datetime]) -> Dict[datetime.datetime, Dict[str, int]]:
    """
    Returns the float values by day for each stock

    Stocks must be valid stocks. Will result in an error if the name of the stock does not exist
    :param stocks: a list of stock symbols
    :param days: A list of days
    :return: The float values for each stock for each day
    """
    # For now, we will just see if it works for getting the float for today only
    stock = get_stock("AAPL")
    pprint(stock)


if __name__ == '__main__':
    stocks = ["MSFT"]
    days = [datetime.datetime.now()]

    get_float(stocks, days)
