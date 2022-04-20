import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
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

#################################################

# def show_raw_visualization(data):
#     time_data = data[date_time_key]
#     fig, axes = plt.subplots(
#         nrows=5, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
#     )
#     for i in range(len(feature_keys)):
#         key = feature_keys[i]
#         c = colors[i % (len(colors))]
#         t_data = data[key]
#         t_data.index = time_data
#         t_data.head()
#         ax = t_data.plot(
#             ax=axes[i // 2, i % 2],
#             color=c,
#             title="{}".format(titles[i]),
#             rot=25,
#         )
#         ax.legend([titles[i]])
#     plt.tight_layout()
#     plt.savefig('figures/periodicity_sentiment_price_data.png')
#     plt.show()
#
#
# show_raw_visualization(df)
#
# #################################################
#
# def show_heatmap(data):
#     plt.matshow(data.corr())
#     plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
#     plt.gca().xaxis.tick_bottom()
#     plt.yticks(range(data.shape[1]), data.columns, fontsize=14)
#
#     cb = plt.colorbar()
#     cb.ax.tick_params(labelsize=14)
#     plt.title("Feature Correlation Heatmap", fontsize=14)
#     plt.savefig('figures/correlation_heatmap.png')
#     plt.show()
#
#
# show_heatmap(df)

"""Preprocessing"""
# using trailing 30 days of data
# price 3 days into the future will be used as label => 3

split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
step = 6

past = 30
future = 3
learning_rate = 0.001
batch_size = 256
epochs = 10

#################################################


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


print(
    "The selected parameters are:",
    ", ".join([titles[i] for i in [0, 1, 2, 3, 4, 5, 6, 7]]),
)
selected_features = [feature_keys[i] for i in [0, 1, 2, 3, 4, 5, 6, 7]]
features = df[selected_features]
features.index = df[date_time_key]
features.head()

features = normalize(features.values, train_split)
features = pd.DataFrame(features)
features.head()

train_data = features.loc[0: train_split - 1]
val_data = features.loc[train_split:]

#################################################

start = past + future
end = start + train_split

x_train = train_data[[i for i in range(len(features.columns))]].values
y_train = features.iloc[start:end][[1]]

sequence_length = int(past / step)

#################################################

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

#################################################

x_end = len(val_data) - past - future

label_start = train_split + past + future

x_val = val_data.iloc[:x_end][[i for i in range(7)]].values
y_val = features.iloc[label_start:][[1]]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)

#################################################

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

#################################################

path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)

#################################################

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(history, "Training and Validation Loss")