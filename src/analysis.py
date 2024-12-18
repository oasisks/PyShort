news_example1 = (
    f"Recent News "
    f"Summary: \n"
    f"Article 1: Elon Musk has hired a Republican operative with field organizing expertise to assist in his political activities, indicating an increased involvement in Republican politics and voter mobilization efforts. \n"
    f"Article 2: California lawmakers have passed a bill introducing new restrictions on artificial intelligence, which, if signed by Governor Gavin Newsom, could establish a national standard for AI regulation. \n"
    f"Article 3: Tesla’s initiative to open its Supercharger network to other carmakers is advancing slowly, creating uncertainty about when the network will be accessible to non-Tesla vehicles, despite initial enthusiasm from drivers and industry experts. \n"
    f"Keywords: \n"
    f"Elon Musk, Republican politics, voter mobilization, Tesla, Supercharger network, electric cars, California legislation, artificial intelligence regulation, Gavin Newsom, AI restrictions \n \n"
    f"Stock Percentage Change Prediction: -5 \n \n"
    f"Stock Return: Negative \n \n"
    f"Analysis: "
    f"The forecasted negative stock return stems from a combination of news and model predictions. The articles suggest uncertainty, with the political involvement of Elon Musk potentially distracting from Tesla's business focus, and the slow rollout of the Supercharger network could dampen investor confidence. Additionally, the negative regulatory news regarding AI restrictions adds to the sense of risk for Tesla, aligning with the -5% predicted change from the ARIMA and LSTM models, leading to a forecasted decline in the stock price."
)

news_example2 = ("Recent News "
                 "Summary: \n"
                 "Article 1: The Justice Department is considering taking legal action against Google, including forcing the company to divest parts of its business, to address its alleged monopoly in search. While not directly about MSFT, this could have implications for Microsoft's competing search engine, Bing. \n"
                 "Article 2: Microsoft, alongside other big tech companies like Google and Meta, is aggressively integrating AI chatbots into their products. This reflects Microsoft’s focus on leveraging AI to enhance its ecosystem, indicating its strong positioning in the AI race. \n"
                 "Article 3: Cerebras, an AI chip company, is set to go public, potentially expanding investor options in the AI sector. This news highlights Microsoft’s strategic importance as an established player in AI, competing with Nvidia and other emerging companies. \n"
                 "Keywords: \n"
                 "Microsoft, Google, Meta, Apple, artificial intelligence, AI chatbots, monopoly, Bing, Justice Department, AI ecosystem, Cerebras, Nvidia, AI competition, stock market debut. \n \n"
                 "Stock Percentage Change Prediction: -1 \n \n"
                 "Stock Return: Neutral\n \n"
                 "Analysis: "
                 "The forecast for Microsoft’s stock return is neutral due to a balance of positive and negative factors. On the positive side, Microsoft's strong positioning in the AI space, highlighted by its integration of AI chatbots, aligns well with the market's growing focus on AI. Additionally, the potential indirect benefits from the government’s scrutiny of Google could improve Microsoft’s Bing, though the full impact is uncertain. On the other hand, the news about Cerebras’ IPO introduces more competition in AI, and the -1% model prediction suggests that investors are weighing these mixed signals cautiously. As a result, the stock is likely to see a modest, neutral change in price.")

news_example3 = ("Recent News "
                 "Summary: \n"
                 "Article 1: Huawei, Apple's competitor in China, responded strategically to the unveiling of the iPhone 16, highlighting its resilience despite U.S. trade restrictions. This reflects the competitive pressure Apple faces in the Chinese market. \n"
                 "Article 2: The European Union’s highest court ruled against Apple and Google in landmark cases, marking a significant victory for the EU in regulating big tech. This decision could have implications for Apple's business operations and regulatory compliance in the European market. \n"
                 "Article 3: Foldable phones from competitors like Motorola and Google are improving in quality and affordability, offering innovative alternatives to traditional smartphones. This indicates growing competition for Apple as consumers explore new smartphone designs. \n"
                 "Keywords: \n"
                 "Apple, iPhone 16, Huawei, China, U.S. trade restrictions, European Union, tech regulation, legal cases, Motorola, Google, foldable phones, bendable screens, smartphone competition. \n \n"
                 "Stock Percentage Change Prediction: 7 \n \n"
                 "Stock Return: Positive \n \n"
                 "Analysis: "
                 "The forecasted positive stock return for Apple stems from a combination of factors. While the competitive pressure from Huawei and the foldable phone market, along with the legal challenges posed by the EU ruling, create some headwinds for Apple, they are not expected to significantly derail the company’s momentum. The launch of the iPhone 16 and Apple's ability to navigate regulatory challenges are expected to drive confidence in the stock. The 7% predicted change from the ARIMA and LSTM models reflects optimism around Apple's growth prospects, leading to a forecasted positive return."
                 )


def generate_news_analysis_prompt(ticker: str, summary: str, prediction: float) -> str:
    """
    A helper function to generate a news analysis prompt for each ticker
    :param ticker: symbol
    :param summary: news summary
    :param prediction: prediction
    :return: Returns the analysis prompt for news
    """
    news_analysis_prompt = (
        f"Instruction: Forecast the next day's stock return (price change) for {ticker}, given some news sources that came out "
        f"recently, keywords, and a predicted percentage change from an ARIMA and LSTM model trained on historical stock data. \n \n"
        f"Recent News: \n {summary} \n \n"
        f"Stock Percentage Change Prediction: {prediction} \n \n"
        f"Few-shot Examples: \n 1. {news_example1} \n \n 2. {news_example2} \n \n 3. {news_example3} \n \n"
        f"Given this information, predict the next opening day's summary, and forecast the stock return as positive "
        f"if the stock's price increases by at least 3%, neutral if the stock's price stays within less than a 3% change, "
        f"or negative if the stock's price decreases by at least 3%. The predicted summary should explain the "
        f"return forecasting. It would be best if you predicted what could happen, not merely summarize the history. "
        f"Please focus on explaining your prediction. Can you give a step-by-step explanation before the finalized output? \n \n"
        f"Use format: \n"
        "Stock Return: {Positive/Neutral/Negative} \n"
        "Analysis: {Explain the reasoning behind the prediction, including relevant news and model outputs}")

    return news_analysis_prompt


tweets_example1 = ("Recent Tweets Summary: \n"
                   "Tweet 1: Questions why TSLA stock dropped suddenly, asking if any news caused the decline. \n"
                   "Tweet 2: Challenges \"dip buyers\" to take action during TSLA's drop, implying skepticism or sarcasm. \n"
                   "Tweet 3: Criticizes Tesla, claiming Elon Musk prioritizes Dogecoin over Tesla, with negative sentiment calling the company \"garbage.\" \n"
                   "Tweet 4: Asks about potential layoffs at Tesla, speculating about job cuts. \n"
                   "Tweet 5: Groups TSLA with other stocks (SPY, NVDA, AMD) while stating that AMD is laying off employees and speculates Tesla will follow due to hype without revenues. \n"
                   "Keywords: \n TSLA, Tesla, drop, dip buyers, Dogecoin, DOGE.X, Elon Musk, layoffs, hype, revenues, SPY, NVDA, AMD, employee layoffs, EV company. \n \n"
                   "Stock Percentage Change Prediction: -5 \n \n"
                   "Stock Return: Negative \n \n"
                   "Analysis: The forecasted negative stock return for Tesla is driven by a combination of factors. The tweets reflect deep skepticism, with criticism of Elon Musk’s priorities, concerns about potential layoffs, and negative speculation about Tesla’s future revenues. This has likely led to a decrease in investor confidence, with the -5% predicted change from the ARIMA and LSTM models aligning with these negative sentiments. Consequently, the stock is expected to experience a significant decline on the next trading day."
                   )

tweets_example2 = ("Recent Tweets Summary: \n"
                   "Tweet 1: Criticizes Microsoft for dependency on its products, referencing a bug in the Windows 11 24H2 update and claiming software issues persist despite its success. \n"
                   "Tweet 2: Highlights Microsoft's announcement of new health-care AI tools aimed at improving efficiency in building applications and saving time for clinicians, portraying it as a strategic move in the AI health sector. \n"
                   "Tweet 3: Jokingly mentions a $420 breakout level for MSFT while referencing unrelated tickers (TLRY, CGC). \n"
                   "Tweet 4: Expresses disappointment over MSFT's performance, suggesting a retail-driven downturn and pessimism regarding $420 call options. \n"
                   "Tweet 5: Observes MSFT's stock price testing key levels and hints at sharing further updates. \n"
                   "Keywords: \n MSFT, Microsoft, Windows 11, bug, patch, AI tools, health-care, clinicians, key levels, breakout, $420, retail shakeout, TLRY, CGC, stock action, software issues. \n \n"
                   "Stock Percentage Change Prediction: 1 \n \n"
                   "Stock Return: Neutral \n \n"
                   "Analysis: The forecast for Microsoft’s stock return is neutral due to a combination of factors. While the announcement of healthcare AI tools positions the company in a strong, growing market and should contribute to positive sentiment, the criticism surrounding Windows 11 bugs and software issues dampens investor enthusiasm. Additionally, the mention of $420 breakout levels and retail-driven downturn introduces some uncertainty and caution. Overall, the 1% predicted change from the ARIMA and LSTM models suggests that Microsoft’s stock will experience slight positive movement, but overall, the return will remain neutral."
                   )

tweets_example3 = ("Recent Tweets Summary: \n"
                   "Tweet 1: N/A - The tweet mixes irrelevant information about an unrelated attack with a disparaging comment on bears. \n"
                   "Tweet 2: The user speculates about the direction of AAPL stock after market participants process new information. \n"
                   "Tweet 3: Highlights iOS 18's new feature that impacts satellite phones, presenting it as an advantage for hikers and backpackers to own iPhones. \n"
                   "Tweet 4: Discusses trading profits from AAPL calls, referencing historical performance and options trading strategies. Mentions optimistic projections tied to SPY and AAPL’s price levels. \n"
                   "Tweet 5: Celebrates AAPL’s continued dominance with positive sentiment, using emojis and affirmations. \n"
                   "Keywords: \n AAPL, Apple, iOS 18, satellite phones, hiker, backpacker, stock, trading, options, "
                   "calls, SPY, $572, profits, FOMC, crown, positive sentiment. \n \n"
                   "Stock Percentage Change Prediction: 4 \n \n"
                   "Stock Return: Positive \n \n"
                   "Analysis: The forecasted positive stock return for Apple is driven by a combination of factors. The satellite phone feature in iOS 18 is a notable innovation that could attract new customers, especially those in specialized markets like hikers and backpackers. The trading profits and optimistic projections surrounding AAPL add to the bullish sentiment, and the 4% predicted change from the ARIMA and LSTM models further suggests that the market is confident in Apple’s short-term performance. Consequently, Apple’s stock is expected to experience a positive return on the next trading day.")


def generate_tweets_analysis_prompt(ticker: str, summary: str, prediction: float) -> str:
    """
    A helper function to generate a tweet summary for each ticker
    :param ticker: symbol
    :param summary: tweets summary
    :param prediction: prediction
    :return: Returns the analysis prompt for the tweets
    """
    tweets_analysis_prompt = (f"Instruction: Forecast the next day’s stock return (price change) for {ticker}, "
                              "given some tweets that came out recently on StockTwits, keywords, and a predicted" 
                              "percentage change from an ARIMA and LSTM model trained on historical stock data. \n \n"
                              f"Recent Tweets: \n {summary} \n \n"
                              f"Stock Percentage Change Prediction: {prediction} \n \n"
                              f"Few-shot Examples: \n 1. {tweets_example1} \n \n 2. {tweets_example2} \n \n 3. {tweets_example3} \n \n"
                              f"Given this information, predict the next opening day's summary, and forecast the stock return as positive "
                              f"if the stock's price increases by at least 3%, neutral if the stock's price stays within less than a 3% change, "
                              f"or negative if the stock's price decreases by at least 3%. The predicted summary should explain the "
                              f"return forecasting. It would be best if you predicted what could happen, not merely summarize the history. "
                              f"Please focus on explaining your prediction. Can you give a step-by-step explanation before the finalized output? \n \n"
                              f"Use format: \n"
                              "Stock Return: {Positive/Neutral/Negative} \n"
                              "Analysis: {Explain the reasoning behind the prediction, including relevant news and model outputs}"
                              )
    return tweets_analysis_prompt


if __name__ == "__main__":
    # Test the functions
    # print(generate_news_analysis_prompt("AAPL", "Apple unveils new iPhone 16", 1))
    print(generate_tweets_analysis_prompt("AAPL", "Apple unveils new iPhone 16", 1))
