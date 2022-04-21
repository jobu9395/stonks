import os
from comet_ml import Experiment

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


# connect to comet to record experiments
load_dotenv()
experiment = Experiment(
    api_key=os.getenv('comet_api_key'),
    project_name=os.getenv('comet_project_name'),
    workspace=os.getenv('comet_workspace'),
)

"""Global variables"""
EPOCHS = 20
N_INPUT = 30  # trailing trading day count
BATCH_SIZE = 32
FILENAME = 'dataset/training_data.csv'
# FILENAME = 'dataset/daily_stock_price_data.csv'

hyperparams = {
    FILENAME,
    EPOCHS,
    N_INPUT,
    BATCH_SIZE,
}

# log hyperparameters
experiment.log_parameter("hyperparameters", hyperparams)
print("connection succesful")

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
    experiment.log_figure(periodicity)
    # plt.show()


def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    heatmap = plt.savefig('figures/correlation_heatmap.png')
    experiment.log_figure(heatmap)
    # plt.show()


def time_series_prediction(df):
    df = df.drop('Date', axis=1)
    print(df.head())
    y_col = 'Close' # define y variable, i.e., what we want to predict

    test_size = int(len(df) * 0.05)
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
        batch_size=BATCH_SIZE
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

    model.fit_generator(generator, epochs=EPOCHS)

    loss_per_epoch = model.history.history['loss']
    plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
    loss_curve = plt.savefig('figures/loss_curve.png')
    experiment.log_figure(loss_curve)
    experiment.log_metric("loss", loss_per_epoch)
    # plt.show()

    X_test = test.drop(y_col, axis=1).copy()
    scaled_X_test = Xscaler.transform(X_test)
    test_generator = TimeseriesGenerator(scaled_X_test, np.zeros(len(X_test)), length=N_INPUT, batch_size=BATCH_SIZE)

    y_pred_scaled = model.predict(test_generator)
    y_pred = Yscaler.inverse_transform(y_pred_scaled)
    results = pd.DataFrame({'y_true': test[y_col].values[N_INPUT:], 'y_pred': y_pred.ravel()})

    plt.figure(figsize=(14, 5))
    results.plot()
    time_series_img = plt.savefig('figures/time_series_predictions.png')
    experiment.log_figure(time_series_img)
    # plt.show()


if __name__ == "__main__":
    show_raw_visualization(df)
    show_heatmap(df)
    time_series_prediction(df)
