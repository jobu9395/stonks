import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset/training_data.csv')

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

################################################

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
    plt.savefig('figures/periodicity_sentiment_price_data.png')
    # plt.show()


def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.savefig('figures/correlation_heatmap.png')
    # plt.show()


def time_series_prediction(df):
    df = df.drop('Date', axis=1)
    print(df.head())
    y_col = 'Close' # define y variable, i.e., what we want to predict

    test_size = int(len(df) * 0.05) # here I ask that the test data will be 10% (0.1) of the entire data
    train = df.iloc[:-test_size, :].copy() # the copy() here is important, it will prevent us from getting: SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.
    # Try using .loc[row_index,col_indexer] = value instead
    test = df.iloc[-test_size:, :].copy()

    X_train = train.drop(y_col,axis=1).copy()
    y_train = train[[y_col]].copy() # the double brakets here are to keep the y in dataframe format, otherwise it will be pandas Series

    Xscaler = MinMaxScaler(feature_range=(0, 1)) # scale so that all the X data will range from 0 to 1
    Xscaler.fit(X_train)
    scaled_X_train = Xscaler.transform(X_train)
    Yscaler = MinMaxScaler(feature_range=(0, 1))
    Yscaler.fit(y_train)
    scaled_y_train = Yscaler.transform(y_train)
    scaled_y_train = scaled_y_train.reshape(-1) # remove the second dimention from y so the shape changes from (n,1) to (n,)

    scaled_y_train = np.insert(scaled_y_train, 0, 0)
    scaled_y_train = np.delete(scaled_y_train, -1)

    n_input = 30 #how many samples/rows/timesteps to look in the past in order to forecast the next sample
    n_features= X_train.shape[1] # how many predictors/Xs/features we have to predict y
    b_size = 32 # Number of timeseries samples in each batch
    generator = TimeseriesGenerator(
        scaled_X_train,
        scaled_y_train,
        length=n_input,
        batch_size=b_size
    )

    model = Sequential()
    model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    model.fit_generator(generator, epochs=50)

    loss_per_epoch = model.history.history['loss']
    plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
    # plt.savefig('figures/loss_curve.png')
    # plt.show()

    X_test = test.drop(y_col,axis=1).copy()
    scaled_X_test = Xscaler.transform(X_test)
    test_generator = TimeseriesGenerator(scaled_X_test, np.zeros(len(X_test)), length=n_input, batch_size=b_size)

    y_pred_scaled = model.predict(test_generator)
    y_pred = Yscaler.inverse_transform(y_pred_scaled)
    results = pd.DataFrame({'y_true':test[y_col].values[n_input:], 'y_pred': y_pred.ravel()})

    results.plot()
    plt.savefig('figures/time_series_predictions.png')
    # plt.show()


if __name__ == "__main__":
    show_raw_visualization(df)
    show_heatmap(df)
    time_series_prediction(df)
