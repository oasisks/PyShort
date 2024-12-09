import time
import json
import pandas as pd

from functions.scraper.Ticker import get_recent_time_series
from ML.LSTM_ARIMA import format_data, make_ARIMA_pred, make_LSTM_pred, mixed_prediction, predicted_movement
from typing import List
from summary import SummaryStatus
from rag import get_summary, retrieve_batch_summarize, generate_analysis, get_ticker_news


def generate_predictions(tickers: List[str], news_filename: str, tweets_filename: str):
    """
    This will generate predictions for a list of stocks and output it to a file
    :param tickers: a set of tickers symbols
    :param news_filename: the output filename for news
    :param tweets_filename: the output file for tweets
    :return: None
    """
    saved_predictions = open("cache.txt", "r", encoding="utf-8")
    predictions = saved_predictions.readlines()
    saved_predictions.close()
    saved_predictions = open("cache.txt", "a", encoding="utf-8")
    ticker_histories = get_recent_time_series(tickers)
    i = 0
    for ticker, history in ticker_histories.items():
        # let us first calculate all the ML things
        print(f"Doing ticker: {ticker}")

        if i < len(predictions):
            i += 1
            print("Skipped")
            continue
        dataset = format_data(history)
        arima_results = make_ARIMA_pred(dataset)
        lstm_results = make_LSTM_pred(dataset)
        mixed_results = mixed_prediction(arima_results, lstm_results)
        percentage = predicted_movement(*mixed_results)["predicted movement"] * 100
        saved_predictions.write(str(percentage) + "\n")
        predictions.append(percentage)
    saved_predictions.close()
    predictions = [float(_.strip("\n")) for _ in predictions]
    print(tickers)
    # all_news = get_ticker_news(tickers)
    # print(all_news)
    # news_batch, tweets_batch = get_summary(tickers)
    #
    # while True:
    #     time.sleep(1)
    #     print("Retrieving summaries")
    #     news_status, news_jsonl = retrieve_batch_summarize(news_batch.id)
    #     tweets_status, tweets_jsonl = retrieve_batch_summarize(tweets_batch.id)
    #
    #     if news_status == SummaryStatus.COMPLETED and tweets_status == SummaryStatus.COMPLETED:
    #         break
    #
    # news_analysis = generate_analysis(news_jsonl, predictions)
    # tweets_analysis = generate_analysis(tweets_jsonl, predictions)
    #
    # while True:
    #     time.sleep(1)
    #     print("Retrieving Analysis")
    #     news_status, news_jsonl = retrieve_batch_summarize(news_analysis.id)
    #     tweets_status, tweets_jsonl = retrieve_batch_summarize(tweets_analysis.id)
    #
    #     if news_status == SummaryStatus.COMPLETED and tweets_status == SummaryStatus.COMPLETED:
    #         break
    #
    # with open(news_filename, "w") as file:
    #     for line in news_jsonl:
    #         file.write(json.dumps(line) + "\n")
    #
    # print(f"Successfully written the news file to: {news_filename}")
    #
    # with open(tweets_filename, "w") as file:
    #     for line in tweets_jsonl:
    #         file.write(json.dumps(line) + "\n")
    #
    # print(f"Successfully written the news file to: {tweets_filename}")


def get_short_results():
    filename = "current_short.txt"
    short_file = open(filename, "r", encoding="utf-8")
    tickers = [ticker.strip("\n") for ticker in short_file]
    short_news = get_ticker_news(tickers)
    news = get_ticker_news(tickers)
    output = {}
    for ticker, _ in news.items():
        output[ticker] = [{"content": article.content, "headline": article.headline} for article in _]

    predictions = []
    ticker_histories = get_recent_time_series(tickers)
    i = 0
    for ticker, history in ticker_histories.items():
        # let us first calculate all the ML things
        print(f"Doing ticker: {ticker}")

        if i < len(predictions):
            i += 1
            print("Skipped")
            continue
        dataset = format_data(history)
        arima_results = make_ARIMA_pred(dataset)
        lstm_results = make_LSTM_pred(dataset)
        mixed_results = mixed_prediction(arima_results, lstm_results)
        percentage = predicted_movement(*mixed_results)["predicted movement"] * 100
        predictions.append(percentage)


if __name__ == '__main__':
    filename = "current_short.txt"
    short_file = open(filename, "r", encoding="utf-8")
    tickers = [ticker.strip("\n") for ticker in short_file]

    print(tickers)

    short_news = get_ticker_news(tickers)
    with open("short_news.json", "w", encoding="utf-8") as file:
        news = get_ticker_news(tickers)
        output = {}
        for ticker, _ in news.items():
            output[ticker] = [{"content": article.content, "headline": article.headline} for article in _]
        file.write(json.dumps(output))
    # tickers = ["CGBS"]
    # generate_predictions(tickers, "short_news.csv", "short_tweets.csv")

    filename = "constituents.csv"
    # market_file = open(filename, "r", encoding="utf-8")

    dataset = pd.read_csv(filename)

    tickers = dataset["Symbol"].tolist()
    with open("long_news.json", "w", encoding="utf-8") as file:
        news = get_ticker_news(tickers)
        output = {}
        for ticker, _ in news.items():
            output[ticker] = [{"content": article.content, "headline": article.headline} for article in _]
        file.write(json.dumps(output))
    print(tickers)
