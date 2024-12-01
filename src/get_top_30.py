import json
import os
from datetime import datetime
import time
from typing import Dict, Set
import heapq


def search_for_date(ticker_symbol: str, date: datetime) -> float:
    """
    Search for the date given a ticker and returns the shares outstanding
    :param ticker_symbol: the ticker symbol
    :param date: the date of
    :return:
    """
    ticker_directory = os.path.join("data", "time_series")
    ticker_symbol = ticker_symbol.capitalize()
    path = os.path.join(ticker_directory, f"{ticker_symbol}.csv")
    if not os.path.exists(path):
        return -1.0

    file = open(path, "r", encoding="utf-8")
    file.readline()
    content = [line.strip("\n") for line in file]

    low = 0
    high = len(content) - 1
    mid = 0
    while low <= high:
        mid = (high + low) // 2

        mid_content = content[mid].split(",")
        mid_date = datetime.strptime(mid_content[0], "%Y-%m-%d")

        if mid_date < date:
            low = mid + 1
        elif mid_date > date:
            high = mid - 1
        else:
            # this means we found it
            return float(mid_content[1])

    return -1.0


def get_shares_outstanding(dates: Set[datetime]) -> Dict[datetime, Dict[str, float]]:
    """
    This return a dictionary per date where each value is a dictionary of stocks with its shares outstanding
    :param dates: the dates
    :return: Returns a dictionary of dates and dictionary of stock to shares outstanding
    """
    output = {}
    file_directory = "data/time_series"
    file_names = [name for name in os.listdir(file_directory)]
    for file_name in file_names:
        ticker = file_name.strip(".csv")
        file = open(os.path.join(file_directory, file_name), "r", encoding="utf-8")
        file.readline()
        for line in file:
            line = line.strip("\n").split(",")
            date = datetime.strptime(line[0], "%Y-%m-%d")
            shares_outstanding = float(line[1])
            price_close = float(line[3])
            if date in dates:
                if date not in output:
                    output[date] = {ticker: (shares_outstanding, price_close)}
                else:
                    output[date][ticker] = (shares_outstanding, price_close)
        file.close()

    return output


def get_top_30_market_cap():
    """
    This goes through all the files in the time series data and find the top 30 largest market cap stocks per day and
    writes it to data
    :return: None
    """
    file_directory = os.path.join("data")

    files = [f for f in os.listdir(file_directory) if os.path.isfile(os.path.join(file_directory, f))][:-2]
    dates = {datetime.strptime(f.strip(".csv"), "%Y-%m-%d") for f in files}
    market_cap_directory = os.path.join("data", "top30marketcap")
    all_shares_outstanding = get_shares_outstanding(dates)

    for date, shares_outstanding in all_shares_outstanding.items():
        formatted_date = date.strftime("%Y-%m-%d")
        file_path = os.path.join(market_cap_directory, f"{formatted_date}.csv")
        file = open(file_path, "w", encoding="utf-8")
        market_caps = [(shares * price, ticker) for ticker, (shares, price) in shares_outstanding.items()]
        top30 = heapq.nlargest(30, market_caps)

        top_30_stocks = [(ticker, cap) for cap, ticker in top30]
        file.write("ticker,marketcap\n")
        for ticker, cap in top_30_stocks:
            file.write(f"{ticker},{cap:.2f}\n")

        file.close()


def get_top_30_short():
    """
    This goes through all the short interest for each day, and then we will put them into a list
    :return: None
    """

    file_directory = "data"
    files = [f for f in os.listdir(file_directory) if os.path.isfile(os.path.join(file_directory, f))][:-2]

    dates = {datetime.strptime(f.strip(".csv"), "%Y-%m-%d") for f in files}
    start_time = time.time()

    all_shares_outstanding = get_shares_outstanding(dates)

    for file_name in files:
        short_percentages = []
        file = open(os.path.join(file_directory, file_name), "r", encoding="utf-8")
        file.readline()
        date = None
        for line in file:
            line = line.strip("\n").split("\",\"")
            date = datetime.strptime(line[-1].lstrip("\"").rstrip("\""), "%Y-%m-%d")
            ticker_symbol = line[1].lstrip("\"").rstrip("\"")
            if ticker_symbol not in all_shares_outstanding[date]:
                continue
            shares_outstanding = all_shares_outstanding[date][ticker_symbol][0]
            shares_floated = float(line[5].lstrip("\"").rstrip("\""))

            short_percent = shares_floated / shares_outstanding
            short_percentages.append((short_percent, ticker_symbol))

        file.close()

        top30 = sorted(short_percentages, key=lambda x: x[0], reverse=True)[:30]
        top30file = open(f"data/top30short/{date.date()}.csv", "w", encoding="utf-8")
        top30file.write(f"percentage,ticker_name\n")
        top30file.writelines([f"{top[0]},{top[1]}\n" for top in top30])
        top30file.close()
    end_time = time.time()

    print(end_time - start_time)


if __name__ == '__main__':
    # get_top_30_short()

    get_top_30_market_cap()
