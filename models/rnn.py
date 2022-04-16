
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

import matplotlib.pyplot as plt


'''This models past numeric only data related to price using an RNN.'''

LAGGING_DAYS_FOR_TRAINING_DATA = 30

df = pd.read_csv(
    'dataset/daily_stock_prices_all_numbers.csv',
    date_parser=True
)

data_training = df[df['Date'] < '2022-03-01'].copy()
data_test = df[df['Date'] >= '2022-03-01'].copy()

training_data = data_training.drop(['Date', 'Adj Close'], axis=1)
test_data = data_test.drop(['Date'], axis=1)

print(f"training data head: \n {training_data.head()}\n")
print(f"test data head: \n {test_data.head()}\n")
print(f"training data described: \n {training_data.describe()}\n")
print(f"test data described: \n {test_data.describe()}\n")

scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)

print(training_data)

X_train = []
y_train = []

for i in range(60, training_data.shape[0]):
    X_train.append(training_data[i-60:i])
    y_train.append(training_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train.shape, y_train.shape)

regressor = Sequential()

regressor.add(
    LSTM(
        units=50,
        activation='relu',
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2]),
    )
)

regressor.add(Dropout(0.2))

regressor.add(
    LSTM(
        units=60,
        activation='relu',
        return_sequences=True,
    )
)
regressor.add(Dropout(0.3))

regressor.add(
    LSTM(
        units=80,
        activation='relu',
        return_sequences=True,
    )
)
regressor.add(Dropout(0.4))

regressor.add(
    LSTM(
        units=120,
        activation='relu',
    )
)
regressor.add(Dropout(0.5))

regressor.add(
    Dense(units=1)
)

print(regressor.summary())

regressor.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

regressor.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32
)


# TODO prepare test dataset
# timestamp 34:04 on YT tutorial

past_60_days = data_training.tail(60)
df = past_60_days.append(data_test, ignore_index=True)
print(df)

df = df.drop(['Date', 'Adj Close'], axis=1)

inputs = scaler.transform(df)
print(inputs)

X_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
print(X_test.shape, y_test.shape)

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



