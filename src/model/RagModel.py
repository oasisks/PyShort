from __future__ import annotations

from pydantic import BaseModel, Field


class SummaryOutput(BaseModel):
    ticker: str = Field(default=None, title="ticker", description="The ticker symbol.")
    news_summary: str = Field(default=None, title="news_summary",
                              description="Summary related to yfinance news articles.")
    tweets_summary: str = Field(default=None, title="tweets_summary",
                                description="Summary related to tweets from stocktweets.")


class Article(BaseModel):
    headline: str = Field(default=None, title="headline", description="The headline of the article.")
    content: str = Field(default=None, title="content", description="The content of the article.")
