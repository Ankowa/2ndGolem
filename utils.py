import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

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
        pd.DataFrame(sequences[:,:,idx]).to_csv(os.path.join('cleaned_data', name), index=None)

def dumpTabular(tabular, isTest=False):
    postfix = 'test' if isTest else 'train'
    tabular.to_csv(os.path.join('cleaned_data', 'X_'+postfix+'.csv'), index=None)

def getDataloaders(X, y, split=0.25, batch_size=10):
    if isinstance(X, pd.DataFrame):
        X = np.array(X)
    X = torch.from_numpy(X).type(torch.double)
    y = torch.from_numpy(y).type(torch.long)
    dataset = TensorDataset(X, y)
    train_dataset, test_dataset = random_split(dataset, split)
    return DataLoader(train_dataset, batch_size, True), DataLoader(test_dataset, batch_size, True)
