import time

import streamlit as st
from typing import Dict, List
from rag import get_summary, generate_analysis
from llm.summary import list_batches, retrieve_batch_summarize, SummaryStatus
from functions.scraper.Ticker import get_recent_time_series
from ML.LSTM_ARIMA import format_data, make_ARIMA_pred, make_LSTM_pred, predicted_movement, mixed_prediction


@st.cache_data
def get_batches():
    return list_batches()


def side_bar():
    # adding the batches
    st.sidebar.title("View Batch Summaries/Analysis")
    batches = {batch[1]: batch[0] for batch in get_batches()}

    menu = st.sidebar.selectbox("Select a batch", [""] + [batch for batch in batches])

    # this will just show the batch results and its summaries
    if menu and not menu.isspace():
        status, response = retrieve_batch_summarize(batches[menu])
        st.sidebar.write("# Retrieved Summaries/Analysis")
        st.sidebar.write(f"## Status : {status}")
        for line in response:
            # st.write(response)
            st.sidebar.write(f"### For {line['custom_id'].split('_')[0].upper()}")
            items = line["response"]["body"]["choices"][0]["message"]["content"].split("\n")
            i = 1
            for item in items:
                if item.startswith("Tweet") or item.startswith("Article"):
                    st.sidebar.write(f"{i}. {item}")
                    i += 1
                else:
                    st.sidebar.write(item)


def get_predictions(tickers: List[str]) -> List[float]:
    """
    A helper function to get a list of percentage change that is parallel to the tickers list
    :param tickers: a set of tickers
    :return:
    """
    predictions = []
    with st.spinner("Generating Predictions. Please wait."):
        for ticker, history in get_recent_time_series(tickers).items():
            # st.write(history)
            dataset = format_data(history)
            arima_results = make_ARIMA_pred(dataset)
            lstm_results = make_LSTM_pred(dataset)
            mixed_results = mixed_prediction(arima_results, lstm_results)
            predictions.append(predicted_movement(*mixed_results)["predicted movement"] * 100)

    return predictions


def main():
    st.set_page_config(
        page_title="PyShort Demo",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"  # Keep the sidebar expanded and non-collapsible
    )

    side_bar()

    st.title("Stock Price Prediction")

    # Add some input for the user
    tickers = st.text_input("Enter stock symbols separated by commas (i.e. AAPL, GOOG)")
    is_valid = True

    tickers = tickers.split(",")
    for ticker in tickers:
        if not ticker:
            is_valid = False
        if ticker.isspace():
            is_valid = False

    if st.button("Submit", key="start") and is_valid:
        ### Context Retriver ###
        # Rag component will grab the last 24 hours of tweets for the given ticker
        # Rag component will grab the last 24 hours of news for the given ticker

        ### Summarizer gpt4 ###
        # Summarizer component will summarize the news and tweets for the given ticker
        with st.spinner("Initializing Request. Please wait."):
            news_batch, tweets_batch = get_summary(tickers)
            get_batches.clear()

        news_jsonl = tweets_jsonl = None
        # we will wait until the batch finishes or times out after 30 seconds
        with st.spinner("Generating Summaries. Please wait."):
            get_batches.clear()
            i = 0
            time_limit = 120
            while i < time_limit:
                news_status, news_jsonl = retrieve_batch_summarize(news_batch.id)
                tweets_status, tweets_jsonl = retrieve_batch_summarize(tweets_batch.id)

                if news_status == SummaryStatus.COMPLETED and tweets_status == SummaryStatus.COMPLETED:
                    # we will now pass this on to the next part
                    break
                else:
                    news_jsonl = None
                    tweets_jsonl = None
                # sleep for one second
                time.sleep(1)
                i += 1

        if news_jsonl is None and tweets_jsonl is None:
            st.write(
                "It seems it is still generating. You can select summary batches below to continue once it finishes")
        else:
            # ML movel will predict the movement of the stock price for the given
            # ticker based on the numerical data from the rag component
            predictions = get_predictions(tickers)

            st.write(f"This is the predictions: {predictions}")
            # Explanation Generator gpt4
            # Explanation component will generate an explanation for the prediction
            # based on the summarized news and tweets and ml model output
            with st.spinner("Initializing request to Analyzer. Please wait."):
                news_analysis_batch = generate_analysis(news_jsonl, predictions)
                tweets_analysis_batch = generate_analysis(tweets_jsonl, predictions)
                get_batches.clear()

            news_analysis_jsonl = tweets_analysis_jsonl = None
            with st.spinner("Generating Analysis"):
                get_batches.clear()
                i = 0
                time_limit = 120
                while i < time_limit:
                    news_analysis_status, news_analysis_jsonl = retrieve_batch_summarize(news_analysis_batch.id)
                    tweets_analysis_status, tweets_analysis_jsonl = retrieve_batch_summarize(tweets_analysis_batch.id)

                    if news_analysis_status == SummaryStatus.COMPLETED and tweets_analysis_status == SummaryStatus.COMPLETED:
                        break
                    else:
                        news_analysis_jsonl = tweets_analysis_jsonl = None
                    time.sleep(1)
                    i += 1
            if news_analysis_jsonl is None and tweets_analysis_jsonl is None:
                st.write(
                    "It seems it is still generating. You can select the analysis from the sidebar at the right when "
                    "it is ready.")

            # st.write(news_jsonl)
            # st.write(tweets_jsonl)

    st.header("Select Summary Batches")
    batches = {batch[1]: batch[0] for batch in get_batches() if batch[1].strip(".json").strip(".jsonl")}

    options = st.multiselect("Select a batch", [batch for batch in batches])

    if st.button("Submit", key="summary_batches"):
        for option in options:
            batch_id = batches[option]
            status, batch = retrieve_batch_summarize(batch_id)
            if status == SummaryStatus.PROCESSING:
                st.write(f"The selected batch: {option} is still processing. Please select it later.")
            elif status == SummaryStatus.ERROR:
                st.write(f"The selected batch: {option} was not successfully handled. Please try again.")
            else:
                # we want to pass the batch summaries of either tweets or news to gpt for analysis
                tickers = [line["custom_id"].split("_")[0] for line in batch]

                predictions = get_predictions(tickers)

                st.write(f"This is the predictions: {predictions}")

                with st.spinner("Generating Analysis. Please wait"):
                    # Explanation Generator gpt4
                    # Explanation component will generate an explanation for the
                    # prediction based on the summarized news and tweets and ml model output
                    analysis_batch = generate_analysis(batch, predictions)
                    get_batches.clear()
                    analysis_jsonl = None
                    i = 0
                    time_limit = 120
                    while i < time_limit:
                        analysis_status, analysis_jsonl = retrieve_batch_summarize(analysis_batch.id)

                        if analysis_status == SummaryStatus.COMPLETED:
                            break
                        else:
                            analysis_jsonl = None
                        time.sleep(1)
                        i += 1
                    if analysis_jsonl is None:
                        st.write(
                            "It seems the analysis is still generating, you can view the analysis on the side bar when it"
                            "is ready."
                        )
    ### Movement Predictor ###


main()
