import streamlit as st

st.title("Stock Price Prediction")

# Add some input for the user
ticker = st.text_input("Enter a stock symbol", "AAPL")

### Context Retriver ###
# Rag component will grab the last 24 hours of tweets for the given ticker
# Rag component will grab the last 24 hours of news for the given ticker
# Rag component will grab the last 24 hours of tweets for the given ticker

### Summarizer gpt4 ###
# Summarizer component will summarize the news and tweets for the given ticker

### Movement Predictor ###
# ML movel will predict the movement of the stock price for the given ticker based on the numerical data from the rag component

### Explanation Generator gpt4 ###
# Explanation component will generate an explanation for the prediction based on the summarized news and tweets and ml model output

# Display the prediction
st.write(f"Prediction for ticker: {ticker}")
