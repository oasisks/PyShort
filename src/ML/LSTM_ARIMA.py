import os
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pylab import rcParams
from tqdm import tqdm
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, LSTM, Bidirectional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ARIMA
def make_ARIMA_pred(dataset, visualize=False):
    def test_stationarity(prices, window):

        rolmean = pd.DataFrame(prices).rolling(window).mean()
        rolstd = pd.DataFrame(prices).rolling(window).std()
        adft = adfuller(prices, autolag='AIC')
        output = pd.Series(adft[0:4],
                           index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
        for key, values in adft[4].items():
            output['critical value (%s)' % key] = values
        return output["p-value"]

    def predict_next_movement(prices):

        prices = np.array(prices)
        window = 12 if len(prices) > 12 else len(prices)
        p_value = test_stationarity(prices, window)
        log_prices = np.log(prices)

        # if p_value > 0.05:
        #     result = seasonal_decompose(log_prices, model='multiplicative', period=12)
        #     rolling_mean = pd.DataFrame(log_prices).rolling(window=window).mean()
        #     std_dev = pd.DataFrame(log_prices).rolling(window=window).std()

        train_data = log_prices[3:]  # offset to remove edge cases

        model_autoARIMA = auto_arima(train_data,
                                     start_p=0,
                                     start_q=0, test='adf',
                                     max_p=4,
                                     max_q=4, m=1,
                                     d=None, seasonal=False, start_P=0,
                                     D=0,
                                     trace=True,
                                     error_action='ignore',
                                     suppress_warnings=True,
                                     stepwise=True)

        model = ARIMA(train_data, order=model_autoARIMA.get_params()["order"])
        fitted = model.fit()
        prediction = fitted.forecast(1, alpha=0.05)  # 1 day-ahead prediction, 95% conf
        return prediction

    prices = dataset["price_close"]
    previous_price = prices.iloc[-2]
    current_date = dataset.index[-1]
    current_price = prices.iloc[-1]

    prediction = predict_next_movement(prices[:-1])
    predicted_current_price = prediction[0]

    if visualize:
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.autoscale(axis='x', tight=True)
        plt.xticks([i for i in range(0, len(dataset), 60)], rotation=90)

        plt.plot(dataset[:-1], label='Price')
        plt.scatter(current_date, current_price, s=1, label="actual")
        plt.scatter(current_date, np.exp(predicted_current_price), s=1, label="predicted")

        plt.title("Close Price Movement")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    return (previous_price, current_price, np.exp(predicted_current_price))


# LSTM
def make_LSTM_pred(dataset, visualize=False):
    def prepare_data(data):

        Ms = MinMaxScaler()
        transformed_data = Ms.fit_transform(data)
        training_size = round(len(transformed_data) * 0.8)

        train_data = transformed_data[:training_size]
        test_data = transformed_data[training_size:]

        return train_data, test_data, Ms

    def create_sequences(data, window):

        sequences, labels = [], []
        start_idx = 0

        for stop_idx in range(window, len(data)):
            sequences.append(data[start_idx:stop_idx])
            labels.append(data[stop_idx])
            start_idx += 1

        return np.array(sequences), np.array(labels)

    def make_and_fit_model(train_seq, train_label, test_seq, test_label):

        model = Sequential()
        model.add(
            LSTM(50, activation='relu', return_sequences=True, input_shape=(train_seq.shape[1], train_seq.shape[2])))

        model.add(Dropout(0.1))
        model.add(LSTM(units=50))

        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

        model.fit(train_seq, train_label, epochs=80, validation_data=(test_seq, test_label), verbose=1)

        return model

    data = dataset
    current_date = data.index[-1]
    current_price = data["price_close"].iloc[-1]
    previous_price = data["price_close"].iloc[-2]

    train_data, test_data, Ms = prepare_data(data)
    window = 50 if len(data) > 50 else len(data) // 2

    train_seq, train_label = create_sequences(train_data, window)
    test_seq, test_label = create_sequences(test_data, window)

    model = make_and_fit_model(train_seq, train_label, test_seq, test_label)
    predicted = model.predict(test_seq)
    inverse_predicted = Ms.inverse_transform(predicted)

    if visualize:
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.autoscale(axis='x', tight=True)
        plt.xticks([i for i in range(0, len(dataset), 60)], rotation=90)

        plt.plot(data[:-2], label='Actual')
        plt.scatter(current_date, current_price, label='Actual', s=1)
        plt.scatter(current_date, inverse_predicted[-1], label='Predicted', s=1)

        plt.title("Close Price Movement")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    return (previous_price, current_price, inverse_predicted[-1][0])


def format_data(data):
    data.reset_index(inplace=True)
    data = data.rename(columns={"Date": "date", "Close": "price_close"})
    df = data.dropna()
    df = df[["date", "price_close"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date")
    df = df.set_index('date')
    return df


def predicted_movement(previous, actual_current, predicted_current):
    percent_change = ((actual_current - previous) / previous)
    predicted_percent_change = ((predicted_current - previous) / previous)
    return {"actual movement": percent_change,
            "predicted movement": predicted_percent_change,
            "accuracy": 1 if percent_change * predicted_percent_change > 0 else 0,
            "SE": (percent_change - predicted_percent_change) ** 2}


def mixed_prediction(ARIMA_results, LSTM_results, w_A=0.8, w_L=0.2):
    ARIMA_pred, LSTM_pred = ARIMA_results[-1], LSTM_results[-1]
    return ARIMA_results[0], ARIMA_results[1], w_A * ARIMA_pred + w_L * LSTM_pred


if __name__ == "__main__":
    # file_path = "C:/Users/hbori/Documents/NLP final project/ML/time_series-20241204T012459Z-001/time_series/AA.csv"
    file_path = "./AA.csv"
    dataset = format_data(pd.read_csv(file_path))

    print("ARIMA Prediction")
    ARIMA_results = make_ARIMA_pred(dataset, False)
    print(predicted_movement(*ARIMA_results))

    print("LSTM Prediction")
    LSTM_results = make_LSTM_pred(dataset, False)
    print(predicted_movement(*LSTM_results))

    print("Mixed Prediction")
    mixed_results = mixed_prediction(ARIMA_results, LSTM_results)
    print(predicted_movement(*mixed_results))
