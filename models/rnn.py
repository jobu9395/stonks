import pandas as pd
import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf


'''This models word vectors on a same day relationship to price.'''

df = pd.read_csv('dataset/training_data.csv')

training_data = df[df['date'] < '2021-04-01'].copy()
test_data = df[df['date'] >= '2021-04-01'].copy()

print(training_data)

scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)




