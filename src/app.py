import streamlit as st
from rag import get_summary
from llm.summary import list_batches

st.set_page_config(
    page_title="Non-Collapsible Sidebar",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"  # Keep the sidebar expanded and non-collapsible
)

st.title("Stock Price Prediction")

# adding the batches
st.sidebar.title("Scrollable Sidebar")
batches = list_batches()

menu = st.sidebar.selectbox("Select an option", ["Home", "About", "Contact"])

# Add some input for the user
tickers = st.text_input("Enter stock symbols separated by commas (i.e. AAPL, GOOG)", "AAPL").split(",")

is_valid = True
for ticker in tickers:
    if not ticker:
        is_valid = False

# Create buttons in the sidebar
if st.sidebar.button("Button 1"):
    st.write("You clicked Button 1!")

if st.sidebar.button("Button 2"):
    st.write("You clicked Button 2!")
# for i in range(1, 101):  # Simulating a large number of items
#     st.sidebar.write(f"Item {i}")
### Context Retriver ###
# Rag component will grab the last 24 hours of tweets for the given ticker
# Rag component will grab the last 24 hours of news for the given ticker

### Summarizer gpt4 ###
# Summarizer component will summarize the news and tweets for the given ticker
# get_summary(tickers)

### Movement Predictor ###
# ML movel will predict the movement of the stock price for the given ticker based on the numerical data from the rag component

### Explanation Generator gpt4 ###
# Explanation component will generate an explanation for the prediction based on the summarized news and tweets and ml model output

# Display the prediction

if is_valid:
    st.write(f"Prediction for ticker: {tickers}")
else:
    st.write("Invalid syntax.")
