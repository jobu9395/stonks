from comet_ml import Experiment

import os

import torch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, input_size, lagging_days, hidden_size, dropout):
        super().__init__()
        self.input_size = input_size
        self.lagging_days = lagging_days
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=4,
                            dropout=self.dropout)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(self.lagging_days * self.hidden_size, 1)

    def forward(self, input):
        lstm_out, (h, c) = self.lstm(input)
        flatten = self.flat(lstm_out)
        linear = self.linear(flatten)
        return linear


TRAINING_DATA_FILE = 'dataset/training_data.csv'
TRAIN_SPLIT_DATE_STRING = '2022-03-01'

hyper_params = {
    'lagging_days': 10,
    'dropout': 0.1,
    'batch_size': 32,
    'epochs': 30,
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
    data_training = df[df['Date'] < TRAIN_SPLIT_DATE_STRING].copy()
    data_test = df[df['Date'] >= TRAIN_SPLIT_DATE_STRING].copy()

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
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

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
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    return X_train, y_train, X_test, y_test


def train(epoch, X_train, y_train, model, optimizer, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    for idx, (data, target) in enumerate(zip(X_train, y_train)):
        data = torch.unsqueeze(data, dim=0)
        target = torch.unsqueeze(target, dim=0)
        
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
                  .format(epoch, idx, len(list(zip(X_train, y_train))), iter_time=iter_time, loss=losses, top1=acc))
            
    return losses, acc


def validate(epoch, X_test, y_test, model, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # evaluation loop
    for idx, (data, target) in enumerate(zip(X_test, y_test)):
        data = torch.unsqueeze(data, dim=0)
        target = torch.unsqueeze(target, dim=0)
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
                  .format(epoch, idx, len(list(zip(X_test, y_test))), iter_time=iter_time, loss=losses, top1=acc))

    return losses, acc


def main():
    experiment = Experiment(
        api_key = os.getenv('api_key'),
        project_name = os.getenv('project_name'),
        workspace = os.getenv('workspace')
    )
    print("Logging Hyperparameters")
    experiment.log_parameters(hyper_params)
    # pre train stuff goes here
    print("Getting data ready...")
    X_train, y_train, X_test, y_test = pre_train()
    print("Data prep complete!")

    input_size = X_train.shape[2]
    model = RedditStockModel(input_size=input_size, 
                             lagging_days=hyper_params['lagging_days'],
                             hidden_size=32, 
                             dropout=0.1)
    
    optimizer = torch.optim.Adam(model.parameters(), hyper_params['learning_rate'])
    loss_fn = torch.nn.MSELoss()

    print("Begin Training!")
    for epoch in range(hyper_params['epochs']):

        train_losses, train_acc = train(epoch, X_train, y_train, model, optimizer, loss_fn)
        val_losses, val_acc = validate(epoch, X_test, y_test, model, loss_fn)
        
        experiment.log_metric("train loss", train_losses.avg, epoch=epoch)
        experiment.log_metric("train acc", train_acc.avg, epoch=epoch)
        experiment.log_metric("val loss", val_losses.avg, epoch=epoch)
        experiment.log_metric("val acc", val_acc.avg, epoch=epoch)
        
    model.eval()
    with torch.no_grad():
        
        y_pred = model(X_test)
        y_pred = y_pred.numpy()
        
        print(y_pred.shape)
        
        y_test = y_test.numpy()
        
        """Visualization"""

        plt.figure(figsize=(14, 5))
        plt.plot(y_test, color='red', label='Real AMC Stock Price')
        plt.plot(y_pred, color='blue', label='Predicted AMC Stock Price')
        plt.title('AMC Stock Price Prediction using LSTM neural network')
        plt.xlabel('Time')
        plt.ylabel('AMC Stock Price')
        plt.legend()
        experiment.log_figure(figure=plt)
        plt.show()

    # experiment.end()
