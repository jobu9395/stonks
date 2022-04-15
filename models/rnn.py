
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


'''This models past numeric only data related to price using an RNN.'''

# df = pd.read_csv('dataset/training_data.csv')
df = pd.read_csv(
    'dataset/daily_stock_prices_all_numbers.csv',
    date_parser=True
)

training_data = df[df['Date'] < '2022-03-01'].copy()
test_data = df[df['Date'] >= '2022-03-01'].copy()

training_data = training_data.drop(['Date', 'Adj Close'], axis=1)
test_data = test_data.drop(['Date'], axis=1)

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
