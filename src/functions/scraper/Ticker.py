import os
import pandas as pd
import requests

import requests
import yfinance as yf
from datetime import datetime
from devtools import pprint
from typing import List, Dict
from finviz.main_func import get_stock
import matplotlib.pyplot as plt


def get_recent_time_series(stocks: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Returns the float values by day for each stock

    Stocks must be valid stocks. Will result in an error if the name of the stock does not exist
    :param stocks: a list of stock symbols

    :return: The float values for each stock for each day
    """
    # For now, we will just see if it works for getting the float for today only
    tickers = yf.Tickers(" ".join(stocks))

    return {ticker: ticker_obj.history(start="2019-01-01", period="max") for ticker, ticker_obj in
            tickers.tickers.items()}


if __name__ == '__main__':
    # stocks = ["PLCE", "AAPL"]
    # for ticker, history in get_recent_time_series(stocks).items():
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(history.index, history["Close"], label='Close Price', linewidth=2)
    #
    #     plt.title(f"{ticker} Stock Price Over Time", fontsize=16)
    #     plt.xlabel('Date', fontilsize=12)
    #     plt.ylabel('Price (USD)', fontsize=12)
    #     plt.grid()
    #     plt.legend()
    #
    #     plt.show()

    arima_market = "market_performance_LSTM.csv"
    arima_short = "short_performance_LSTM.csv"

    market = open(arima_market, "r", encoding="utf-8")
    short = open(arima_short, "r", encoding="utf-8")

    market.readline()
    short.readline()

    epsilon = 0.01
    correct = 0
    total = 0
    categories = ["Short"]
    percentages = []
    for file in [short]:
        correct = 0
        total = 0
        for line in file:
            index, yesterday, today, predicted = line.strip("\n").split(",")
            yesterday, today, predicted = float(yesterday), float(today), float(predicted[1:-1])

            real_delta = today - yesterday
            pred_delta = predicted - yesterday + epsilon

            if real_delta > 0 and pred_delta > 0:
                correct += 1
            total += 1

        print(correct, total, correct / total)
        percentages.append(correct / total * 100)

    categories = ['Short', 'Market']
    colors = ['#FFE8A3', '#E4CCFF']  # Blue and Orange

    plt.figure(figsize=(10, 6))
    plt.bar(categories, percentages, color=colors, edgecolor='black', width=0.6)

    # Add titles and labels
    plt.title('Correct Predictions by Category (ARIMA)', fontsize=16, pad=20)
    plt.xlabel('Category', fontsize=14, labelpad=10)
    plt.ylabel('Percentage Correct', fontsize=14, labelpad=10)

    # Add gridlines for the y-axis
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add annotations on top of each bar
    for i, v in enumerate(percentages):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=12, color='black')

    # Adjust y-axis limit to avoid clutter
    plt.ylim(0, 100)

    # Make layout clean
    plt.tight_layout()

    # Display the graph
    plt.show()
