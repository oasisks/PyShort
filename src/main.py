import os
from functions.scraper.scraper import get_short_data


def main():
    folder_path = os.path.join(os.path.abspath(os.curdir), "data")

    relevant_dates = ["2024-07-31"]
    missing_dates = []

    for date in relevant_dates:
        file_path = os.path.join(folder_path, f"{date}.txt")
        if not os.path.exists(file_path):
            missing_dates.append(date)

    get_short_data(short_dates=missing_dates)

if __name__ == '__main__':
    pass