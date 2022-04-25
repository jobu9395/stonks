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
from keras.layers import LSTM, Dropout, Flatten
import warnings
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from dotenv import load_dotenv
warnings.filterwarnings('ignore')


load_dotenv()
experiment = Experiment(
        api_key = os.getenv('api_key'),
        project_name = os.getenv('project_name'),
        workspace = os.getenv('workspace')
)

"""Global variables"""
EPOCHS = 10
LEARNING_RATE = 0.0005
N_INPUT = 30 # trailing trading day count
BATCH_SIZE = 128
TEST_SIZE = 0.05
FILENAME = 'dataset/embeddings_training_data.csv'

hyperparams = {
    FILENAME,
    EPOCHS,
    LEARNING_RATE,
    N_INPUT,
    BATCH_SIZE,
}

# log hyperparameters
experiment.log_parameter("hyperparameters", hyperparams)
emb_df = pd.read_csv(FILENAME)


date_time_key = "Date"


################################################
def create_embeddings(data):
    train_ds = tf.data.Dataset.from_tensor_slices(data)
    vectorize_layer = TextVectorization(
        max_tokens=5000,
        output_mode='int',
        output_sequence_length=100
    )
    vectorize_layer.adapt(train_ds.batch(64))

    embed_model = Sequential([
        vectorize_layer,
        Embedding(5000, 150, input_length=N_INPUT),
        Flatten()
    ])

    return embed_model(data)

def time_series_prediction(df):
    df = df.drop('Date', axis=1)
    y_col = 'Close' # define y variable, i.e., what we want to predict

    test_size = int(len(df) * TEST_SIZE) # here I ask that the test data will be 10% (0.1) of the entire data
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
    X_train = create_embeddings(X_train)
    print(X_train.shape, y_train.shape)
    
    n_features= X_train.shape[1] # how many predictors/Xs/features we have to predict y
    b_size = 32 # Number of timeseries samples in each batch
    generator = TimeseriesGenerator(
        X_train,
        y_train,
        length=N_INPUT,
        batch_size=BATCH_SIZE
    )
    print(generator)
    X_test = test.drop(y_col,axis=1).copy()
    X_test = X_test['comment_body'].astype(str).to_numpy(dtype='str')
    
    test_generator = TimeseriesGenerator(
        X_test,
        np.zeros(len(X_test)),
        length=N_INPUT,
        batch_size=BATCH_SIZE)


    regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)
    model = Sequential()
    model.add(LSTM(
        units=50, 
        activation='relu', 
        input_shape=(N_INPUT, n_features), 
        return_sequences=True)
    )
    model.add(Dropout(0.1)),
    model.add(LSTM(
        units=50, 
        activation='relu', 
        input_shape=(N_INPUT, n_features), 
        return_sequences=True)
    )
    model.add(Dropout(0.2)),
    model.add(LSTM(
        units=50, 
        activation='relu', 
        input_shape=(N_INPUT, n_features), 
        return_sequences=True)
    )
    model.add(Dropout(0.3)),
    model.add(LSTM(
        units=50, 
        activation='relu', 
        input_shape=(N_INPUT, n_features), 
        return_sequences=False)
    )
    model.add(Dropout(0.4)),
    # model.add(GlobalMaxPool1D())
    model.add(Dense(1, activity_regularizer=regularizer))
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    print(model.summary())

    history = model.fit(
        generator,
        validation_data=test_generator,
        epochs=EPOCHS)

    # loss_per_epoch = model.history.history['loss']
    # plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
    # plt.savefig('figures/embeddings_loss_curve.png')
    # plt.show()


    y_pred = model.predict(test_generator)
    results = pd.DataFrame({'y_true':test[y_col].values[N_INPUT:], 'y_pred': y_pred.ravel()})

    plot_df = pd.DataFrame.from_dict({'loss': history.history['loss'], 'val_loss': history.history['val_loss']})
    plot_df.plot()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('figures/embeddings_loss_curve.png')
    plt.show()
    plt.close()

    results.plot()
    plt.savefig('figures/embeddings_time_series_predictions.png')
    plt.show()
    plt.close()

def train_model():
    time_series_prediction(emb_df)

if __name__ == "__main__":
    train_model()