import os
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')


"""Global variables"""
EPOCHS = 10
N_INPUT = 30  # trailing trading day count
BATCH_SIZE = 32
TEST_SIZE = 0.05
# FILENAME = 'dataset/training_data.csv'
FILENAME = 'dataset/training_data_all_comments.csv'
# FILENAME = 'dataset/daily_stock_price_data.csv'

# read in data
df = pd.read_csv(FILENAME)

titles = [
    'neg',
    'neu',
    'pos',
    'compound',
    'Open',
    'High',
    'Low',
    'Volume',
    'Close'
]

feature_keys = [
    'neg',
    'neu',
    'pos',
    'compound',
    'Open',
    'High',
    'Low',
    'Volume',
    'Close'
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]

date_time_key = "Date"


def show_raw_visualization(data):
    time_data = data[date_time_key]
    fig, axes = plt.subplots(
        nrows=5, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{}".format(titles[i]),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()
    periodicity = plt.savefig('figures/periodicity_sentiment_price_data.png')
    plt.show()
    plt.close()


def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    heatmap = plt.savefig('figures/correlation_heatmap.png')
    plt.show()
    plt.close()


def time_series_prediction(df):
    df = df.drop('Date', axis=1)
    print(df.head())
    y_col = 'Close' # define y variable, i.e., what we want to predict

    test_size = int(len(df) * TEST_SIZE)
    train = df.iloc[:-test_size, :].copy()
    test = df.iloc[-test_size:, :].copy()

    X_train = train.drop(y_col, axis=1).copy()
    y_train = train[[y_col]].copy()

    Xscaler = MinMaxScaler(feature_range=(0, 1))
    Xscaler.fit(X_train)
    scaled_X_train = Xscaler.transform(X_train)
    Yscaler = MinMaxScaler(feature_range=(0, 1))
    Yscaler.fit(y_train)
    scaled_y_train = Yscaler.transform(y_train)
    scaled_y_train = scaled_y_train.reshape(-1)

    scaled_y_train = np.insert(scaled_y_train, 0, 0)
    scaled_y_train = np.delete(scaled_y_train, -1)

    n_features = X_train.shape[1]
    generator = TimeseriesGenerator(
        scaled_X_train,
        scaled_y_train,
        length=N_INPUT,
        batch_size=BATCH_SIZE,
    )

    model = Sequential(
        [
            LSTM(
                units=50,
                activation='relu',
                return_sequences=True,
                input_shape=(N_INPUT, n_features)
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

    print(model.summary())

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
    )

    X_test = test.drop(y_col, axis=1).copy()
    scaled_X_test = Xscaler.transform(X_test)

    test_generator = TimeseriesGenerator(
        scaled_X_test,
        np.zeros(len(X_test)),
        length=N_INPUT,
        batch_size=BATCH_SIZE
    )

    history = model.fit(
        generator,
        validation_data=test_generator,
        epochs=EPOCHS,
    )

    y_pred_scaled = model.predict(test_generator)
    y_pred = Yscaler.inverse_transform(y_pred_scaled)

    # used for time series plot
    results = pd.DataFrame({'y_true': test[y_col].values[N_INPUT:], 'y_pred': y_pred.ravel()})

    # used for loss curves (validation and training)
    plot_df = pd.DataFrame.from_dict({'loss': history.history['loss'], 'val_loss': history.history['val_loss']})
    plot_df.plot()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('figures/loss_curve.png')
    plt.show()
    plt.close()

    results.plot()
    plt.savefig('figures/time_series_predictions.png')
    plt.show()
    plt.close()


def train_model():
    show_raw_visualization(df)
    show_heatmap(df)
    time_series_prediction(df)


if __name__ == "__main__":
    train_model()
