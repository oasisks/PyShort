from finra_handler import post_request, get_partitions
from compare import CompareFilter
from devtools import pprint


def get_short_data():
    """
    :return:
    """
    group = "otcMarket"
    dataset = "consolidatedShortInterest"
    partitions = get_partitions(group, dataset)
    most_recent_short_date = partitions.partitions[0].values[0]
    # payload = {
    #     "compareFilters": [
    #         CompareFilter().equals().field_name("marketClassCode").value("NYSE").filter,
    #     ]
    # }
    #
    # response = post_request(group, dataset, payload)
    #
    # data = response.content.decode().strip("\n").split("\n")
    #
    # for row in data:
    #     print(row)


if __name__ == '__main__':
    get_short_data()