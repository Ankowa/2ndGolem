import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import AdamW
from time import time
from sklearn.metrics import accuracy_score

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
    for X, y in loader:
        pred = model(X)
        loss = crit(pred, y.reshape(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()

def test(model, loader, crit):
    with torch.no_grad():
        full_loss = 0
        count = 0
        accuracy = 0
        for X, y in loader:
            pred = model(X)
            loss = crit(pred, y)
            full_loss += loss.item()
            count += len(y)
            accuracy = accuracy_score(pred, y) * len(y)
        print('Loss:', loss/count, 'accuracy:', accuracy/count) 

def train(model, train_dataloader, test_dataloader, epochs, show_every=10, save=False, name=None):
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    start = time()
    for epoch in range(1, epochs+1):
        train_once(model, train_dataloader, optimizer, criterion)
        if not epoch%show_every:
            print('_'*32)
            print('epoch:', epoch, 'time:', round(time()-start, 2))
            test(model, test_dataloader, criterion)
    if save:
        torch.save(model.state_dict(), os.path.join('models', name))
    
