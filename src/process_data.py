import os
from functions.scraper.scraper import get_short_data


def any_empty(*args):
    return any(not var for var in args)


def main():
    filename = "data/raw_pricedata.csv"
    file = open(filename, "r", encoding="utf-8")
    headers = file.readline()
    current_ticker = None
    ticker_file = None
    file_directory = os.path.join("data", "time_series")
    new_headers = [
        "date",
        "shares_outstanding",
        "trading_volume",
        "price_close",
        "price_high",
        "price_low",
        "price_open"
    ]
    total_datapoints = 0
    missing_datapoints = 0
    threshold = 0.01

    # this is the minimum amount of data that a file should have or else it is insufficient
    data_memory_kb = 25
    for line in file:
        line = line.strip("\n").split(",")
        (gvkey, iid, date, ticker_name, conm, shares_outstanding, trading_volume, price_close, price_high, price_low,
         price_open) = line
        if ticker_name == "PRN":
            continue
        if current_ticker is None:
            current_ticker = ticker_name
            ticker_file = open(os.path.join(file_directory, f"{ticker_name}.csv"), "w", encoding="utf-8")
            ticker_file.write(f"{','.join(new_headers)}\n")
        else:
            if current_ticker != ticker_name:
                # we need to determine if we want to delete the file or not base off missing data
                ticker_file.close()
                if missing_datapoints / total_datapoints > threshold:
                    os.remove(os.path.join(file_directory, f"{current_ticker}.csv"))

                missing_datapoints = 0
                total_datapoints = 0
                current_ticker = ticker_name

                ticker_file = open(os.path.join(file_directory, f"{ticker_name}.csv"), "w", encoding="utf-8")
                ticker_file.write(f"{','.join(new_headers)}\n")

        total_datapoints += 1

        if any_empty(date, shares_outstanding, trading_volume, price_close, price_high, price_low, price_open):
            missing_datapoints += 1
            continue

        ticker_file.write(
            f"{','.join([date, shares_outstanding, trading_volume, price_close, price_high, price_low, price_open])}\n"
        )

    if missing_datapoints / total_datapoints > threshold:
        os.remove(os.path.join(file_directory, f"{current_ticker}.csv"))

    ticker_file.close()


if __name__ == '__main__':
    main()
