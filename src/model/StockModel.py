from __future__ import annotations
from pydantic import BaseModel, Field


class Stock(BaseModel):
    name: str = Field(default=None, title="name")
    stockInfo: StockInfo = Field(default=None, title="stockInfo")


class StockInfo(BaseModel):
    symbol: str = Field(default=None, title="symbol")
    companyName: str = Field(default=None, title="companyName")
    primaryExchange: str = Field(default=None, title="primaryExchange")
    sector: str = Field(default=None, title="sector")
    calculationPrice: str = Field(default=None, title="calculationPrice")
    open: float = Field(default=None, title="open")
    openTime: int = Field(default=None, title="openTime")
    close: float = Field(default=None, title="close")
    closeTime: int = Field(default=None, title="closeTime")
    high: float = Field(default=None, title="high")
    low: float = Field(default=None, title="low")
    latestPrice: float = Field(default=None, title="latestPrice")
    latestSource: str = Field(default=None, title="latestSource")
    latestTime: str = Field(default=None, title="latestTime")
    latestUpdate: int = Field(default=None, title="latestUpdate")
    latestVolume: int = Field(default=None, title="latestVolume")
    iexRealtimePrice: float = Field(default=None, title="iexRealtimePrice")
    iexRealtimeSize: int = Field(default=None, title="iexRealtimeSize")
    iexLastUpdated: int = Field(default=None, title="iexLastUpdated")
    delayedPrice: float = Field(default=None, title="delayedPrice")
    delayedPriceTime: int = Field(default=None, title="delayedPriceTime")
    oddLotDelayedPrice: float = Field(default=None, title="oddLotDelayedPrice")
    oddLotDelayedPriceTime: int = Field(default=None, title="oddLotDelayedPriceTime")
    extendedPrice: float = Field(default=None, title="extendedPrice")
    extendedChange: float = Field(default=None, title="extendedChange")
    extendedChangePercent: float = Field(default=None, title="extendedChangePercent")
    extendedPriceTime: int = Field(default=None, title="extendedPriceTime")
    previousClose: float = Field(default=None, title="previousClose")
    previousVolume: int = Field(default=None, title="previousVolume")
    change: float = Field(default=None, title="change")
    changePercent: float = Field(default=None, title="changePercent")
    volume: int = Field(default=None, title="volume")
