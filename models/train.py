from comet_ml import Experiment

import os

import torch
import time
import pandas as pd
import numpy as np

from torch import nn
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

load_dotenv()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class RedditStockModel(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=1, 
                            hidden_size=self.hidden_size, 
                            num_layers=4,
                            dropout=self.dropout)
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        lstm_out, (h, c) = self.lstm(input)
        linear = self.linear(lstm_out)
        return linear


TRAINING_DATA_FILE = 'dataset/training_data.csv'
TRAIN_SPLIT_DATE_STRING = '2022-03-01'

hyper_params = {
    'lagging_days': 10,
    'dropout': 0.1,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.1
}


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc

def pre_train():
    df = pd.read_csv(
        TRAINING_DATA_FILE,
        date_parser=True
    )
    data_training = df[df['Date'] < hyper_params['lagging_days']].copy()
    data_test = df[df['Date'] >= hyper_params['lagging_days']].copy()

    training_data = data_training.drop(['Date'], axis=1)
    test_data = data_test.drop(['Date'], axis=1)

    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)
    # experiment.log_dataset_hash(training_data)

    X_train = []
    y_train = []

    for i in range(hyper_params['lagging_days'], training_data.shape[0]):
        X_train.append(training_data[i-hyper_params['lagging_days']:i])
        y_train.append(training_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    past_training_days = data_training.tail(hyper_params['lagging_days'])
    df = past_training_days.append(data_test, ignore_index=True)

    df = df.drop(['Date'], axis=1)
    inputs = scaler.transform(df)

    X_test = []
    y_test = []

    for i in range(hyper_params['lagging_days'], inputs.shape[0]):
        X_test.append(inputs[i-hyper_params['lagging_days']:i])
        y_test.append(inputs[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_train, y_train, X_test, y_test

def train(epoch, X_train, y_train, model, optimizer, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    for idx, (data, target) in enumerate(zip(X_train, y_train)):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        out = model(data)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        batch_acc = accuracy(out, target)

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 50 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                  .format(epoch, idx, len(zip(X_train, y_train)), iter_time=iter_time, loss=losses, top1=acc))

def validate(epoch, X_test, y_test, model, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # evaluation loop
    for idx, (data, target) in enumerate(zip(X_test, y_test)):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target).item()

        batch_acc = accuracy(out, target)

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t')
                  .format(epoch, idx, len(zip(X_test, y_test)), iter_time=iter_time, loss=losses, top1=acc))

    return acc.avg

def main():
    experiment = Experiment(
        api_key = os.getenv('api_key'),
        project_name = os.getenv('project_name'),
        workspace = os.getenv('workspace')
    )
    experiment.log_parameters(hyper_params)
    # pre train stuff goes here
    X_train, y_train, X_test, y_test = pre_train()

    model = RedditStockModel(hidden_size=32, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), hyper_params['learning_rate'])
    loss_fn = torch.nn.MSELoss()

    for epoch in range(hyper_params['epochs']):

        train(epoch, X_train, y_train, model, optimizer, loss_fn)
        acc = validate(epoch, X_test, y_test, model, optimizer, loss_fn)

    experiment.end()
