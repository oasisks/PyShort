import aiohttp
import asyncio
from datetime import datetime
import csv
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta


def write_to_csv(data, filename, fieldnames=['id', 'body', 'created_at']):
    """
    Write a list of dictionaries to a CSV file.
    @param data: list of dictionaries to write to the CSV file
    @param filename: str, name of the CSV file to write to

    @return: None
    """
    # Open the CSV file for writing
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header (column names)
        writer.writeheader()

        # Write each dictionary in the data list as a row
        for row in data:
            writer.writerow(row)


def read_from_csv(filename):
    data = []
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert each row (an OrderedDict) to a regular dictionary
            data.append(dict(row))
    return data


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


async def fetch_all_messages_for_stock(session, symbol, start_date, end_date, save_tweets=False, max_id=None):
    """
    Fetch all messages for a given stock symbol within a specified date range.
    @param session: aiohttp.ClientSession object
    @param symbol: str, stock symbol to fetch messages for
    @param start_date: str, start date in ISO format
    @param end_date: str, end date in ISO format
    @param save_tweets: bool, optional parameter to save tweets to CSV files
    @param max_id: int, optional parameter to fetch older messages

    @return: list of messages for the given stock symbol within the date range
    """
    all_messages = []
    start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
    end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

    current_date = end_date
    current_messages = []
    while True:
        messages = await fetch_messages(session, symbol, max_id)
        if not messages:
            break

        for msg in messages:
            created_at = datetime.fromisoformat(msg['created_at'].replace("Z", "+00:00"))

            if start_date <= created_at <= end_date:
                all_messages.append(msg)
                current_messages.append(msg)
            elif created_at < start_date:
                if save_tweets:
                    write_to_csv(current_messages,
                                 f"{symbol}_{created_at.strftime('%Y-%m-%d %H:%M:%S')}_{current_date.strftime('%Y-%m-%d %H:%M:%S')}.csv")

                return all_messages  # Stop if messages are older than the start date

            if save_tweets and current_date - created_at >= timedelta(days=180):
                write_to_csv(current_messages,
                             f"{symbol}_{created_at.strftime('%Y-%m-%d %H:%M:%S')}_{current_date.strftime('%Y-%m-%d %H:%M:%S')}.csv")
                current_date = created_at
                current_messages = []

        max_id = messages[-1]['id']  # Update max_id to fetch older messages in next call

        await asyncio.sleep(1)  # Add a delay to avoid hitting rate limits

    return all_messages


async def fetch_all_stocks(symbols_to_grab, save_tweets=False, max_id_dict={}):
    """
    Fetch all messages for a list of stock symbols within a specified date range.
    @param symbols_to_grab: dict, stock symbols and their corresponding start_date and end_date
    @param save_tweets: bool, optional parameter to save tweets to CSV files
    @param max_id_dict: dict, optional parameter to store max_id for each stock symbol

    @return: dictionary of stock symbols and their corresponding messages
    """
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_all_messages_for_stock(session, symbol, dates["start_date"], dates["end_date"],
                                         save_tweets=save_tweets, max_id=max_id_dict.get(symbol, None))
            for symbol, dates in symbols_to_grab.items()
        ]
        results = await asyncio.gather(*tasks)
        return {symbol: messages for symbol, messages in zip(symbols_to_grab.keys(), results)}


# Example usage
# symbols = [
#     "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "NVDA", "META", "TSLA", "AVGO", "PEP", 
#     "COST", "ADBE", "NFLX", "CMCSA", "TXN", "AMD", "AMGN", "INTC", "HON", "QCOM",
#     "INTU", "PYPL", "CSCO", "AMAT", "SBUX", "MDLZ", "ISRG", "ADP", "BKNG", "MU", 
#     "LRCX", "ADI", "ATVI", "VRTX", "REGN", "ZM", "MRNA", "ASML", "SNPS", "KLAC",
#     "MAR", "GILD", "TEAM", "MNST", "CTSH", "ROST", "MELI", "NXPI", "EA", "DOCU", 
#     "EXC", "ILMN", "JD", "FTNT", "CRWD", "WDAY", "KDP", "CTAS", "BIIB", "ABNB",
#     "CEG", "ORLY", "PANW", "WBA", "FAST", "AEP", "SGEN", "CHTR", "VRSK", "BIDU", 
#     "CSX", "ODFL", "DXCM", "PCAR", "MRVL", "DDOG", "PAYX", "CPRT", "OKTA", "ZS",
#     "MTCH", "LULU", "CDNS", "BMRN", "NTES", "ALGN", "IDXX", "PDD", "DD", "TTD", 
#     "NTAP", "SPLK", "SIRI", "FISV", "TTWO", "SWKS", "ANSS", "TSCO", "FLT", "CHKP"
# ]

# symbols = ["TSLA"]
# start_date = "2024-11-13T00:00:00Z"
# end_date = "2024-11-14T00:00:00Z"
# max_id_dict = {}  # Optional max_id for each stock symbol

# symbols_to_grab = {
#     "TSLA": {"start_date": "2024-11-13T00:00:00Z", "end_date": "2024-11-14T00:00:00Z"},
#     "AAPL": {"start_date": "2024-09-18T00:00:00Z", "end_date": "2024-09-19T00:00:00Z"},
#     "MSFT": {"start_date": "2024-10-10T00:00:00Z", "end_date": "2024-10-11T00:00:00Z"},
# }

# Run the asynchronous fetching
async def main():
    symbols_to_grab = {
        "TSLA": {"start_date": "2024-11-13T00:00:00Z", "end_date": "2024-11-14T00:00:00Z"},
        "AAPL": {"start_date": "2024-09-18T00:00:00Z", "end_date": "2024-09-19T00:00:00Z"},
        "MSFT": {"start_date": "2024-10-10T00:00:00Z", "end_date": "2024-10-11T00:00:00Z"},
    }
    max_id_dict = {}
    stock_messages = await fetch_all_stocks(symbols_to_grab, max_id_dict=max_id_dict)
    for symbol, messages in stock_messages.items():
        print(f"\nMessages for {symbol}:")
        for msg in messages:
            print(msg)

# Run the main async function
# asyncio.run(main())

# msg = read_from_csv("/Users/nhung/Desktop/PyShort Project/PyShort/src/MSFT_2024-11-09 00:00:00+00:00_2024-11-09 23:59:59+00:00.csv")
# for m in msg:
#     print(m)
#     print("\n")
