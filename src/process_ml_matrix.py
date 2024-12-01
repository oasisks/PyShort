from datetime import datetime, timedelta
import os

from typing import Dict, List


def process_file(file_directory: str) -> Dict[datetime, List[str]]:
    """
    Processes a file and returns its time series as a dictionary. Assumes a csv file from data/time_series
    :param file_directory: the file_directory
    :return: Dictionary containing the time series information
    """
    file = open(file_directory, "r", encoding="utf-8")
    file.readline()
    time_series_data = {}
    for line in file:
        line = line.strip("\n").split(",")
        date = datetime.strptime(line[0], "%Y-%m-%d")
        time_series_data[date] = line[1:]

    file.close()
    return time_series_data


def process_ml_short(file_name: str) -> None:
    """
    Processes short data
    :param file_name: the file name
    :return: None
    """
    ml_dataset = os.path.join("data", "ml_dataset")
    top30_short_directory = os.path.join("data", "top30short")

    # the short interest
    file = [os.path.join(top30_short_directory, file_dir) for file_dir in os.listdir(top30_short_directory)]
    cached = {}
    top_30_short_ml_set = open(os.path.join(ml_dataset, file_name), "w", encoding="utf-8")
    top_30_short_ml_set.write(",".join(
        ["Symbol", "Date", "Shares_Outstanding", "Trading_Volume", "Price_Close", "Price_High", "Price_Low",
         "Price_Open",
         "Short_Percentage", "Percent_Change"]) + "\n")
    time_series_directory = os.path.join("data", "time_series")

    for i in range(1, len(file)):
        previous = file[i - 1]
        current = file[i]

        previous_date = datetime.strptime(previous.split("\\")[-1].strip(".csv"), "%Y-%m-%d")
        current_date = datetime.strptime(current.split("\\")[-1].strip(".csv"), "%Y-%m-%d")

        previous_file = open(previous, "r", encoding="utf-8")
        previous_file.readline()
        for line in previous_file:
            short_percentage, ticker_name = line.strip("\n").split(",")
            ticker_file_path = os.path.join(time_series_directory, f"{ticker_name}.csv")
            assert os.path.exists(
                ticker_file_path), f"This shouldn't happen. File name: {ticker_name}, Date: {previous}"

            # we want to then cache it
            # the current implementation assumes infinite memory, though my computer has enough memory to support it
            if ticker_name not in cached:
                cached[ticker_name] = process_file(ticker_file_path)

            # assume the ticker is in cached
            date_i = previous_date
            date_i_prev = date_i + timedelta(days=-1)
            while date_i < current_date:
                if date_i_prev not in cached[ticker_name]:
                    date_i_prev = date_i
                    date_i += timedelta(days=1)
                    continue

                if date_i not in cached[ticker_name]:
                    date_i += timedelta(days=1)
                    continue

                previous_day_price = float(cached[ticker_name][date_i_prev][2])
                current_day_price = float(cached[ticker_name][date_i][2])
                price_percent_delta = (current_day_price - previous_day_price) / previous_day_price

                top_30_short_ml_set.write(
                    ",".join(
                        [ticker_name, date_i.strftime("%Y-%m-%d")] + cached[ticker_name][date_i] +
                        [short_percentage, str(price_percent_delta)]) +
                    "\n")

                date_i_prev = date_i
                date_i += timedelta(days=1)


def process_ml_market_cap(file_name: str) -> None:
    """
    Processes market cap folder
    :param file_name: the file name
    :return: None
    """
    ml_dataset = os.path.join("data", "ml_dataset")
    top30_market_directory = os.path.join("data", "top30marketcap")

    # the short interest
    file = [os.path.join(top30_market_directory, file_dir) for file_dir in os.listdir(top30_market_directory)]
    cached = {}
    top_30_short_ml_set = open(os.path.join(ml_dataset, file_name), "w", encoding="utf-8")
    top_30_short_ml_set.write(",".join(
        ["Symbol", "Date", "Shares_Outstanding", "Trading_Volume", "Price_Close", "Price_High", "Price_Low",
         "Price_Open",
         "Short_Percentage", "Percent_Change"]) + "\n")
    time_series_directory = os.path.join("data", "time_series")

    for i in range(1, len(file)):
        previous = file[i - 1]
        current = file[i]

        previous_date = datetime.strptime(previous.split("\\")[-1].strip(".csv"), "%Y-%m-%d")
        current_date = datetime.strptime(current.split("\\")[-1].strip(".csv"), "%Y-%m-%d")

        previous_file = open(previous, "r", encoding="utf-8")
        previous_file.readline()
        for line in previous_file:
            ticker_name, market_cap = line.strip("\n").split(",")
            ticker_file_path = os.path.join(time_series_directory, f"{ticker_name}.csv")
            assert os.path.exists(
                ticker_file_path), f"This shouldn't happen. File name: {ticker_name}, Date: {previous}"

            # we want to then cache it
            # the current implementation assumes infinite memory, though my computer has enough memory to support it
            if ticker_name not in cached:
                cached[ticker_name] = process_file(ticker_file_path)

            # assume the ticker is in cached
            date_i = previous_date
            date_i_prev = date_i + timedelta(days=-1)
            while date_i < current_date:
                if date_i_prev not in cached[ticker_name]:
                    date_i_prev = date_i
                    date_i += timedelta(days=1)
                    continue

                if date_i not in cached[ticker_name]:
                    date_i += timedelta(days=1)
                    continue

                previous_day_price = float(cached[ticker_name][date_i_prev][2])
                current_day_price = float(cached[ticker_name][date_i][2])
                price_percent_delta = (current_day_price - previous_day_price) / previous_day_price

                top_30_short_ml_set.write(
                    ",".join(
                        [ticker_name, date_i.strftime("%Y-%m-%d")] + cached[ticker_name][date_i] +
                        ["0", str(price_percent_delta)]) +
                    "\n")

                date_i_prev = date_i
                date_i += timedelta(days=1)


def main():
    """
    Creates the big matrix used for training, testing, and validation for our ML component

    It will create two csv files where the columns are:
        Ticker Symbol, Date, Shares Outstanding, Trading_Volume, Price Close, Price High, Price Low, Price Open,
         and Short Interest Percentage

    The csvs are created from the files listed in data/top30marketcap and data/top30short
    :return: None
    """

    process_ml_short("short_dataset.csv")
    process_ml_market_cap("market_cap_dataset.csv")


if __name__ == '__main__':
    main()
