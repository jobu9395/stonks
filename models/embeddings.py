import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, TextVectorization, Embedding, GlobalMaxPool1D
from keras.layers import LSTM
import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset/training_data.csv')


date_time_key = "Date"

################################################
def text_vectorization_dataset(dataset):
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    return dataset
    

def time_series_prediction(df):
    df = df.drop('Date', axis=1)
    y_col = 'Close' # define y variable, i.e., what we want to predict

    test_size = int(len(df) * 0.05) # here I ask that the test data will be 10% (0.1) of the entire data
    train = df.iloc[:-test_size, :].copy() # the copy() here is important, it will prevent us from getting: SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.
    # Try using .loc[row_index,col_indexer] = value instead
    test = df.iloc[-test_size:, :].copy()

    X_train = train.drop(y_col,axis=1).copy()
    y_train = train[[y_col]].copy() # the double brakets here are to keep the y in dataframe format, otherwise it will be pandas Series

    y_train = y_train.to_numpy()
    y_train = y_train.reshape(-1) # remove the second dimention from y so the shape changes from (n,1) to (n,)

    y_train = np.insert(y_train, 0, 0)
    y_train = np.delete(y_train, -1)

    train_ds = tf.data.Dataset.from_tensor_slices(X_train)
    vectorize_layer = TextVectorization(
        max_tokens=1000,
        output_mode='int',
        output_sequence_length=100
    )
    vectorize_layer.adapt(train_ds)

    n_input = 30 #how many samples/rows/timesteps to look in the past in order to forecast the next sample
    n_features= X_train.shape[1] # how many predictors/Xs/features we have to predict y
    b_size = 32 # Number of timeseries samples in each batch
    generator = TimeseriesGenerator(
        X_train,
        y_train,
        length=n_input,
        batch_size=b_size
    )

    model = Sequential()
    model.add(Embedding(1000, 150))
    model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features), return_sequences=True))
    model.add(GlobalMaxPool1D())
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
