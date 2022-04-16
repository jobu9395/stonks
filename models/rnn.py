import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

'''This models past numeric only data related to price using an RNN.'''

# global vars
TRAINING_DATA_FILE = 'dataset/daily_stock_price_data.csv'
TRAIN_SPLIT_DATE_STRING = '2022-03-01'
LAGGING_DAYS_FOR_TRAINING_DATA = 10
BATCH_SIZE = 32
EPOCH_COUNT = 5


def train_model():
    df = pd.read_csv(
        TRAINING_DATA_FILE,
        date_parser=True
    )

    data_training = df[df['Date'] < TRAIN_SPLIT_DATE_STRING].copy()
    data_test = df[df['Date'] >= TRAIN_SPLIT_DATE_STRING].copy()

    training_data = data_training.drop(['Date'], axis=1)
    test_data = data_test.drop(['Date'], axis=1)

    print(f"training data head: \n {training_data.head()}\n")
    print(f"test data head: \n {test_data.head()}\n")
    print(f"training data described: \n {training_data.describe()}\n")
    print(f"test data described: \n {test_data.describe()}\n")

    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)

    X_train = []
    y_train = []

    for i in range(LAGGING_DAYS_FOR_TRAINING_DATA, training_data.shape[0]):
        X_train.append(training_data[i-LAGGING_DAYS_FOR_TRAINING_DATA:i])
        y_train.append(training_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    """Builds custom model"""
    regressor = Sequential(
        [
            LSTM(
                units=50,
                activation='relu',
                return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2])
            ),
            Dropout(0.1),
            LSTM(
                units=60,
                activation='relu',
                return_sequences=True,
            ),
            Dropout(0.09),
            LSTM(
                units=70,
                activation='relu',
                return_sequences=True,
            ),
            Dropout(0.08),
            LSTM(
                units=80,
                activation='relu',
            ),
            Dropout(0.07),
            Dense(
                units=1
            )
        ]
    )

    print(regressor.summary())

    regressor.compile(
        optimizer='adam',
        loss='mean_squared_error',
    )

    regressor.fit(
        X_train,
        y_train,
        epochs=EPOCH_COUNT,
        batch_size=BATCH_SIZE,
    )

    past_training_days = data_training.tail(LAGGING_DAYS_FOR_TRAINING_DATA)
    df = past_training_days.append(data_test, ignore_index=True)

    df = df.drop(['Date'], axis=1)
    inputs = scaler.transform(df)

    X_test = []
    y_test = []

    for i in range(LAGGING_DAYS_FOR_TRAINING_DATA, inputs.shape[0]):
        X_test.append(inputs[i-LAGGING_DAYS_FOR_TRAINING_DATA:i])
        y_test.append(inputs[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)

    # normalize
    y_pred = regressor.predict(X_test)
    scale = 1 / (scaler.scale_[0])

    y_pred = y_pred * scale
    y_test = y_test * scale

    """Visualization"""

    plt.figure(figsize=(14, 5))
    plt.plot(y_test, color='red', label='Real AMC Stock Price')
    plt.plot(y_pred, color='blue', label='Predicted AMC Stock Price')
    plt.title('AMC Stock Price Prediction using LSTM neural network')
    plt.xlabel('Time')
    plt.ylabel('AMC Stock Price')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_model()
