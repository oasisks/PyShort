import json
import os

import requests
import time

from errors.finra_errors import DateError
from .finra_handler import post_request, get_partitions
from .filters import CompareFilter, DomainFilters
from devtools import pprint
from loguru import logger
from typing import List
from .utils import check_date_exists


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

    if short_dates != "recent":
        invalid_dates = []
        for date in short_dates:
            invalid_dates.append(date) if not date or not check_date_exists(date, dates) else None

        if invalid_dates:
            raise DateError(message=f"The following dates are invalid: {invalid_dates}")
    else:
        short_dates = [partitions.partitions[0].values[0]]

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

    i = 0
    finished_jobs = 0
    while finished_jobs != len(response_urls):
        i = i % len(response_urls)
        date, url = response_urls[i]
        response = requests.get(url)

        if response.status_code == 202:
            time.sleep(0.1)
            i += 1
            continue
        elif response.status_code == 200:
            result_link = response.json()["resultLink"]

            result_response = requests.get(result_link)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_path = os.path.join(folder_path, f"{date}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result_response.text)

            finished_jobs += 1
