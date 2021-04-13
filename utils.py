import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import AdamW
from time import time
from sklearn.metrics import accuracy_score
from IPython.display import clear_output
import matplotlib.pyplot as plt

def loadTabular(cleaned=False, isTest=False):
    appendix = 'data' if not cleaned else 'cleaned_data'
    if isTest:
        return pd.read_csv(os.path.join(appendix, 'X_test.csv'))
    else:
        return pd.read_csv(os.path.join(appendix, 'X_train.csv')), pd.read_csv(os.path.join('data', 'y_train.csv'), header=None).values

def loadSequential(cleaned=False, isTest=False):
    appendix = 'data' if not cleaned else 'cleaned_data'
    postfix = 'test' if isTest else 'train'
    names = [first+'_'+dim+'_'+postfix+'.csv' for first in ('body_acc', 'body_gyro', 'total_acc') for dim in ('x', 'y', 'z')]
    size = 2947 if isTest else 7352
    return np.concatenate([pd.read_csv(os.path.join(appendix, name), header=None, index_col=0).values.reshape(size, 128, 1) for name in names], axis=2)

def dumpCleanedSequential(sequences, isTest=False):
    postfix = 'test' if isTest else 'train'
    names = [first+'_'+dim+'_'+postfix+'.csv' for first in ('body_acc', 'body_gyro', 'total_acc') for dim in ('x', 'y', 'z')]
    for idx, name in enumerate(names):
        pd.DataFrame(sequences[:,:,idx]).to_csv(os.path.join('cleaned_data', name), index=None, header=None)

def dumpTabular(tabular, isTest=False):
    postfix = 'test' if isTest else 'train'
    tabular.to_csv(os.path.join('cleaned_data', 'X_'+postfix+'.csv'), index=None)

def getDataloaders(X, y, split=0.25, batch_size=10):
    if isinstance(X, pd.DataFrame):
        X = np.array(X)
    y -= 1
    X = torch.from_numpy(X).type(torch.double)
    y = torch.from_numpy(y).type(torch.long)
    dataset = TensorDataset(X, y)
    sizes =  [int(len(dataset)*(1-split)), int(len(dataset)*split)]
    train_dataset, test_dataset = random_split(dataset, sizes)
    return DataLoader(train_dataset, batch_size, True), DataLoader(test_dataset, batch_size, True)

def train_once(model, loader, optim, crit):
    full_loss = 0
    count = 1
    for X, y in loader:
        pred = model(X)
        loss = crit(pred, y.reshape(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        full_loss += loss.item()
        count += 1
    return full_loss/count

def test(model, loader, crit):
    with torch.no_grad():
        full_loss = 0
        count = 0
        accuracy = 0
        for X, y in loader:
            pred = model(X)
            y = y.reshape(-1)
            loss = crit(pred, y)
            full_loss += loss.item()
            count += 1
            accuracy += accuracy_score(pred.max(1)[1], y) * len(y)
        print('Loss:', full_loss/count, 'accuracy:', accuracy/len(loader.dataset))
    return full_loss/count, accuracy/len(loader.dataset)

def plotPerf(data):
    plt.plot(data['train'], label='train')
    plt.plot(data['val'], label='val')
    plt.plot(data['accuracy'], label='accuracy')
    plt.legend()
    plt.show()

def train(model, train_dataloader, test_dataloader, epochs, show_every=10, save=False, name=None):
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    start = time()
    training_performance = {key: [] for key in ('train', 'val', 'accuracy')}
    for epoch in range(1, epochs+1):
        loss = train_once(model, train_dataloader, optimizer, criterion)
        training_performance['train'].append(loss)
        if not epoch%show_every:
            clear_output()
            print('epoch:', epoch, 'time:', round(time()-start, 2))
            loss, accuracy = test(model, test_dataloader, criterion)
            training_performance['val'] += [loss]*show_every
            training_performance['accuracy'] += [accuracy]*show_every
            plotPerf(training_performance)
    if save:
        torch.save(model.state_dict(), os.path.join('models', name))
    
def dumpSubmission(pred, name):
    ids = [_ for _ in range(len(pred))]
    pd.DataFrame({'id': ids, 'Category': pred}).to_csv(os.path.join('submissions', name), index=None)