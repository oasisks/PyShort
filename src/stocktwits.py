import asyncio

import requests
import time

def get_stocktwits_messages(symbol, max_id=None):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    params = {'max': max_id} if max_id else {}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        return response.json().get('messages', [])
    else:
        print("Error:", response.status_code)
        return []

def fetch_all_messages(symbol, start_date, end_date):
    all_messages = []
    max_id = None

    while True:
        messages = get_stocktwits_messages(symbol, max_id)
        if not messages:
            break

        # Filter messages by date range
        for msg in messages:
            print(msg["body"])
            created_at = msg['created_at']
            if start_date <= created_at <= end_date:
                all_messages.append(msg)
            elif created_at < start_date:
                # Stop fetching if we've reached messages older than start_date
                return all_messages

        # Update max_id to fetch older messages in the next iteration
        max_id = messages[-1]['id']
        time.sleep(1)  # Add a delay to avoid hitting rate limits

    return all_messages

# Example usage
symbol = "AAPL"
start_date = "2024-11-01T00:00:00Z"
end_date = "2024-11-07T23:59:59Z"
messages = fetch_all_messages(symbol, start_date, end_date)

for msg in messages:
    print(msg['body'])
