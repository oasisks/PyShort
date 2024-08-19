from model.RequestModel import Partition


def check_date_exists(date: str, dates: list[Partition]) -> bool:
    """
    Using binary search, it checks if the date exists within a list of dates

    Assumes that dates is already ordered
    :param date: the date we are checking
    :param dates: a list of dates
    :return: True if date in dates
    """
    left = 0
    right = len(dates)

    while left <= right:
        mid = left + (right - left) // 2

        mid_val = dates[mid].values[0]
        if mid_val == date:
            return True

        if mid_val > date:
            left = mid + 1
        else:
            right = mid - 1

    return False
