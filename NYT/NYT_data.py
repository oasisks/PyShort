import aiohttp
import asyncio

async def fetch_data(session, url, params = None):
    """Asynchronous function to fetch data from a given URL using aiohttp

    Args:
        session (aiohttp.ClientSession): async session object
        url (str): base API without any parameters
        params (dict, optional): parameters to refine search. Defaults to None.

    Returns:
        response.json() or None: JSON response from the API or None if an error occurs
    """
    try:
        async with session.get(url, params = params) as response:
            return await response.json()
    except Exception as e:
        print(e)
        return None


async def main(key, keyword, filter, begin_date, end_date, pages = 5):
    """Asynchronous function to fetch data from the New York Times API

    Args:
        key (str): Personal API key
        keyword (str): keyword to search for in the articles
        filter (str): additional filters to apply to the search
        begin_date (str): date to start searching from. Uses format: YYYYMMDD
        end_date (str): date to stop searching. Uses format: YYYYMMDD
        pages (int, optional): Number of pages of articles to return. Each page has a maximum of 10 articles. Defaults to 5.

    Returns:
        (list[dict]): pages number of dictionaries containing the articles from the API
    """
    
    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    params = {"api-key": key,
            "q": keyword,
            "fq": filter,
            "begin_date": begin_date,
            "end_date": end_date,
            "sort": "newest",
            "facet_filter": "true"} 
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data(session, url, params|{"page":page}) for page in range(pages)]
        results = await asyncio.gather(*tasks) # runs tasks concurrently

    return [r["response"] for r in results if r]

# example usage
if __name__ == "__main__":
    key = "VAdCJtv9j3REySbXbGoXc5KGcdevjcSO"
    query = "Amazon" 
    # filter = 'source:("The New York Times")' 
    filter = ""
    begin_date = "20240822"
    end_date = "20240922"
    pages = 3

    articles = asyncio.run(main(key, query, filter, begin_date, end_date, pages))
    for page in articles:
        for article in page["docs"]:
            print("Snippet:", article["snippet"])
            print("Lead paragraph:", article["lead_paragraph"])
            # print(article["pub_date"])
            print("\n")
