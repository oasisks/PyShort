import json

from errors.finra_errors import DateError
from finra_handler import post_request, get_partitions
from filters import CompareFilter, DomainFilters
from devtools import pprint
from typing import List
from utils import check_date_exists


def get_short_data(markets: List[str] = None, short_dates: List[str] | str = "recent"):
    """
    Grabs the short interest data from Finra. NOTE: Due to Finra API constraints some ranges might not be available.

    Assumes that the short_dates are valid short_dates. If they are not valid then an error will be raised

    # TODO Make it so that it doesn't matter
    Assumes that the short date is of the format YYYY-MM-DD
    :param markets: A list of markets. Defaults to ["NYSE", "NNM"].
    :param short_dates: The dates we extract short data from. Default to the most recent date.
    :return:
    """
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

        response_urls.append(response.status_link)

    i = 0
    while response_urls:
        i = i % len(response_urls)
        url = response_urls[i]


if __name__ == '__main__':
    short_dates = ["2024-07-31", "2024-07-15"]
    get_short_data(short_dates=short_dates)
