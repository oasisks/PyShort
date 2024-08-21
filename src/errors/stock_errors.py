class StockSymbolError(Exception):
    """
    Exception raised for invalid stock names

    """
    def __init__(self, stock_symbol: str):
        super().__init__(f"The stock symbol '{stock_symbol}' is not valid.")
