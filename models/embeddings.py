from comet_ml import Experiment

import os
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
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

# df = pd.read_csv('dataset/training_data.csv')
emb_df = pd.read_csv('dataset/embeddings_training_data.csv')


date_time_key = "Date"

################################################

def time_series_prediction(df):
    experiment = Experiment(
        api_key = os.getenv('api_key'),
        project_name = os.getenv('project_name'),
        workspace = os.getenv('workspace')
    )
      
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

    train_ds_np = X_train['comment_body'].astype(str).to_numpy(dtype='str')
    X_train = train_ds_np
    train_ds = tf.data.Dataset.from_tensor_slices(train_ds_np)
    vectorize_layer = TextVectorization(
        max_tokens=100,
        output_mode='int',
        output_sequence_length=100
    )
    vectorize_layer.adapt(train_ds.batch(64))

    print(X_train.shape, y_train.shape)
    n_input = 1 #how many samples/rows/timesteps to look in the past in order to forecast the next sample
    n_features= 1 # how many predictors/Xs/features we have to predict y
    b_size = 32 # Number of timeseries samples in each batch
    generator = TimeseriesGenerator(
        X_train,
        y_train,
        length=n_input,
        batch_size=b_size
    )

    model = Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype='string'))
    model.add(vectorize_layer)
    model.add(Embedding(100, 150, input_length=n_input))
    model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features), return_sequences=True))
    model.add(GlobalMaxPool1D())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    model.fit_generator(generator, epochs=15)

    loss_per_epoch = model.history.history['loss']
    plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
    plt.savefig('figures/embeddings_loss_curve.png')
    # plt.show()

    X_test = test.drop(y_col,axis=1).copy()
    X_test = X_test['comment_body'].astype(str).to_numpy(dtype='str')
    test_generator = TimeseriesGenerator(X_test, np.zeros(len(X_test)), length=n_input, batch_size=b_size)

    y_pred = model.predict(test_generator)
    results = pd.DataFrame({'y_true':test[y_col].values[n_input:], 'y_pred': y_pred.ravel()})

    results.plot()
    plt.savefig('figures/embeddings_time_series_predictions.png')
    # plt.show()


if __name__ == "__main__":
    # show_raw_visualization(df)
    # show_heatmap(df)
    time_series_prediction(df)