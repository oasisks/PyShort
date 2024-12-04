import time

import streamlit as st
from rag import get_summary
from llm.summary import list_batches, retrieve_batch_summarize, SummaryStatus


@st.cache_data
def get_batches():
    return list_batches()


def side_bar():
    # adding the batches
    st.sidebar.title("View Batch Summaries")
    batches = {batch[1]: batch[0] for batch in get_batches()}

    menu = st.sidebar.selectbox("Select a batch", [""] + [batch for batch in batches])

    # this will just show the batch results and its summaries
    if menu and not menu.isspace():
        status, response = retrieve_batch_summarize(batches[menu])
        st.sidebar.write("# Retrieved Summaries")
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
                    st.sidebar.text(item)


def main():
    st.set_page_config(
        page_title="Non-Collapsible Sidebar",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"  # Keep the sidebar expanded and non-collapsible
    )

    side_bar()

    st.title("Stock Price Prediction")

    # Add some input for the user
    tickers = st.text_input("Enter stock symbols separated by commas (i.e. AAPL, GOOG)")
    is_valid = True

    typed_nothing = False
    if tickers:
        tickers = tickers.split(",")
        for ticker in tickers:
            if not ticker:
                is_valid = False
            if ticker.isspace():
                is_valid = False
    else:
        typed_nothing = True

    if is_valid and not typed_nothing:
        ### Context Retriver ###
        # Rag component will grab the last 24 hours of tweets for the given ticker
        # Rag component will grab the last 24 hours of news for the given ticker

        ### Summarizer gpt4 ###
        # Summarizer component will summarize the news and tweets for the given ticker
        with st.spinner("Initializing Request. Please wait."):
            get_batches.clear()
            news_batch, tweets_batch = get_summary(tickers)

        news_jsonl = tweets_jsonl = None
        # we will wait until the batch finishes or times out after 30 seconds
        with st.spinner("Generating Summaries. Please wait."):
            get_batches.clear()
            i = 0
            time_limit = 60
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
            st.write("It seems it is still generating. Please click retry or submit new tickers.")

        else:
            st.write(news_jsonl)
            st.write(tweets_jsonl)

    st.header("Select Summary Batches")
    batches = {batch[1]: batch[0] for batch in get_batches()}

    options = st.multiselect("Select a batch", [batch for batch in batches])

    for option in options:
        batch_id = batches[option]
        status, batch = retrieve_batch_summarize(batch_id)
        if status == SummaryStatus.PROCESSING:
            st.write(f"The selected batch: {option} is still processing. Please select it later.")
        elif status == SummaryStatus.ERROR:
            st.write(f"The selected batch: {option} was not successfully handled. Please try again.")
        else:
            # we want to pass the batch summaries of either tweets or news to gpt for analysis
            with st.spinner("Generating Analysis. Please wait"):
                pass
            pass
    ### Movement Predictor ###
    # ML movel will predict the movement of the stock price for the given ticker based on the numerical data from the rag component

    ### Explanation Generator gpt4 ###
    # Explanation component will generate an explanation for the prediction based on the summarized news and tweets and ml model output

    # Display the prediction

    if is_valid:
        st.write(f"Prediction for ticker: {tickers}")
    else:
        st.write("Invalid syntax.")


main()
