import asyncio
import aiohttp
import json
import os

import requests
import time

from errors.finra_errors import DateError
from .finra_handler import post_request, get_partitions
from .filters import CompareFilter, DomainFilters
from devtools import pprint
from loguru import logger
from typing import List, Tuple
from .utils import check_date_exists


async def fetch_status(session: aiohttp.ClientSession, url: str, date: str, folder_path: str) -> bool:
    """
    Fetch each individual status url and perform a task

    :param session the current session
    :param url the url we are extracting from
    :param date the date for the request
    :param folder_path folder path we are writing to
    :return: None
    """
    async with session.get(url) as response:
        response_json = await response.json()
        result_link = response_json["resultLink"]

        if result_link:
            async with session.get(result_link) as result_response:
                final = await result_response.text()

                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                file_path = os.path.join(folder_path, f"{date}.csv")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(final)
                return True
        else:
            logger.error("No result link found")
            return False


async def check_status_links(response_urls: List[Tuple[str, str]], folder_path: str):
    """
    Goes through all the response urls asynchronously
    :param response_urls: the response urls
    :param folder_path: the path of the folder
    :return: None
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_status(session, url, date, folder_path) for date, url in response_urls]
        results = await asyncio.gather(*tasks)
        return results


def get_short_data(markets: List[str] = None, short_dates: List[str] | str = "recent"):
    """
    Grabs the short interest data from Finra. NOTE: Due to Finra API constraints some ranges might not be available.

    Assumes that the short_dates are valid short_dates. If they are not valid then an error will be raised.
    However, if short_dates is empty, the function will just return

    # TODO Make it so that it doesn't matter
    Assumes that the short date is of the format YYYY-MM-DD
    :param markets: A list of markets. Defaults to ["NYSE", "NNM"].
    :param short_dates: The dates we extract short data from. Default to the most recent date.
    :return:
    """
    if not short_dates:
        logger.info("NO_SHORT_DATES")
        return

    root_dir = os.path.abspath(os.curdir)
    folder_path = os.path.join(root_dir, "data")

    if not markets:
        markets = ["NYSE", "NNM"]

    group = "otcMarket"
    dataset = "consolidatedShortInterest"
    partitions = get_partitions(group, dataset)
    dates = partitions.partitions

    if short_dates == "all":
        short_dates = [date.values[0] for date in dates]
    elif short_dates == "recent":
        short_dates = [dates[0].values[0]]
    else:
        invalid_dates = []
        for date in short_dates:
            invalid_dates.append(date) if not date or not check_date_exists(date, dates) else None

        if invalid_dates:
            raise DateError(message=f"The following dates are invalid: {invalid_dates}")

    # filters
    domain_filters = [
        DomainFilters().field_name("marketClassCode").field_values(markets).filter,
    ]

    response_urls = []
    for date in short_dates:
        compare_filters = [
            CompareFilter().equals().field_name("settlementDate").value(date).filter
        ]

        payload = {
            "domainFilters": domain_filters,
            "compareFilters": compare_filters,
            "limit": 100000
        }

        response = post_request(group, dataset, payload, use_async=True)
        response_urls.append((date, response.status_link))

    results = asyncio.run(check_status_links(response_urls, folder_path))
    if all(results):
        print("All finished")
    else:
        print("Some didn't work")
