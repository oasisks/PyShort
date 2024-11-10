import aiohttp
import asyncio
from datetime import datetime

async def fetch_messages(session, symbol, max_id=None):
    """
    Fetch messages for a given stock symbol from the StockTwits API.
    @param session: aiohttp.ClientSession object
    @param symbol: str, stock symbol to fetch messages for
    @param max_id: int, optional parameter to fetch older messages

    @return: list of messages for the given stock symbol
    ['id', 'body', 'created_at']    """
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    params = {'max': max_id} if max_id else {}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }  
    
    keys_to_keep = ['id', 'body', 'created_at']
    async with session.get(url, params=params, headers=headers) as response:
        if response.status == 200:
            data = await response.json()
            messages = data.get('messages', [])
            return [{key: msg[key] for key in keys_to_keep} for msg in messages]
        else:
            print(f"Error fetching {symbol}: {response.status}")
            return []

async def fetch_all_messages_for_stock(session, symbol, start_date, end_date):
    """
    Fetch all messages for a given stock symbol within a specified date range.
    @param session: aiohttp.ClientSession object
    @param symbol: str, stock symbol to fetch messages for
    @param start_date: str, start date in ISO format
    @param end_date: str, end date in ISO format

    @return: list of messages for the given stock symbol within the date range
    """
    all_messages = []
    max_id = None
    start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
    end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

    while True:
        messages = await fetch_messages(session, symbol, max_id)
        if not messages:
            break

        for msg in messages:
            created_at = datetime.fromisoformat(msg['created_at'].replace("Z", "+00:00"))
            if start_date <= created_at <= end_date:
                all_messages.append(msg)
            elif created_at < start_date:
                return all_messages  # Stop if messages are older than the start date

        max_id = messages[-1]['id']  # Update max_id to fetch older messages in next call

        await asyncio.sleep(1)  # Add a delay to avoid hitting rate limits

    return all_messages

async def fetch_all_stocks(symbols, start_date, end_date):
    """
    Fetch all messages for a list of stock symbols within a specified date range.
    @param symbols: list of str, stock symbols to fetch messages for
    @param start_date: str, start date in ISO format
    @param end_date: str, end date in ISO format

    @return: dictionary of stock symbols and their corresponding messages
    """
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_all_messages_for_stock(session, symbol, start_date, end_date)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks)
        return {symbol: messages for symbol, messages in zip(symbols, results)}

# Example usage
symbols = ["AAPL", "TSLA", "MSFT"]  # List of stock symbols
start_date = "2024-11-09T00:00:00Z"
end_date = "2024-11-09T23:59:59Z"

# Run the asynchronous fetching
async def main():
    stock_messages = await fetch_all_stocks(symbols, start_date, end_date)
    for symbol, messages in stock_messages.items():
        print(f"\nMessages for {symbol}:")
        for msg in messages:
            print(msg)

# Run the main async function
asyncio.run(main())
