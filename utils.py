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
from sklearn.ensemble import RandomForestClassifier
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
        print('Loss:', round(full_loss/count, 5), 'accuracy:', round(accuracy/len(loader.dataset), 5))
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
    if not isinstance(pred, np.ndarray):
        pred = pred.max(1)[1]
    if np.unique(pred).max() == 5 or np.unique(pred).min() == 0:
        pred += 1
    ids = [_ for _ in range(len(pred))]
    pd.DataFrame({'id': ids, 'Category': pred}).to_csv(os.path.join('submissions', name), index=None)

def useSeqTab(seq_model, tab_model, sequences, tabular, labels):
    sequences = torch.from_numpy(sequences).type(torch.double)
    try:
        _ = seq_model.features_vector(sequences[0,:,:].reshape(1, 128, 9))
    except Exception as e:
        print(e)
        print('No features vector returning method implemented, EXITTING')
        return
    mask = np.random.random(sequences.shape[0]) > 0.25
    seq_train = sequences[mask]
    tab_train = tabular[mask]
    y_train = labels[mask]
    mask = np.logical_not(mask)
    seq_test = sequences[mask]
    tab_test = tabular[mask]
    y_test = labels[mask]
    train_features_vector = seq_model.features_vector(seq_train)
    test_features_vector = seq_model.features_vector(seq_test)
    X_train = np.concatenate([train_features_vector.detach().numpy(), np.array(tab_train)], axis=1)
    X_test = np.concatenate([test_features_vector.detach().numpy(), np.array(tab_test)], axis=1)
    tab_model.fit(X_train, y_train.ravel())
    pred = tab_model.predict(X_test)
    print('Join model accuracy:', round(accuracy_score(y_test, pred), 5))
    tab_model.fit(X_test, y_test.ravel())
    print('Trained on full data')
    return tab_model

def getJoinPred(seq_model, join_model, sequences, tabular):
    sequences = torch.from_numpy(sequences).type(torch.double)
    features_vector = seq_model.features_vector(sequences)
    data = np.concatenate([features_vector.detach().numpy(), np.array(tabular)], axis=1)
    return join_model.predict(data)

def testCleanedTabular(X_train, X_test, y_train, y_test):
    model = RandomForestClassfier()
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    print('Cleaned dataset accuracy score:', round(accuracy_score(predict, y_test)))
    
