import numpy as np
import pandas as pd
import os

def loadTabular(cleaned=False, isTest=False):
    appendix = 'data' if not cleaned else 'cleaned_data'
    if isTest:
        return pd.read_csv(os.path.join(appendix, 'X_test.csv'))
    else:
        return pd.read_csv(os.path.join(appendix, 'X_train.csv')), pd.read_csv(os.path.join(appendix, 'y_train.csv'), header=None)

def loadSequential(cleaned=False, isTest=False):
    appendix = 'data' if not cleaned else 'cleaned_data'
    postfix = 'test' if isTest else 'train'
    names = [first+'_'+dim+'_'+postfix+'.csv' for first in ('body_acc', 'body_gyro', 'total_acc') for dim in ('x', 'y', 'z')]
    size = 2947 if isTest else 7352
    return np.concatenate([pd.read_csv(os.path.join(appendix, name), header=None, index_col=0).values.reshape(size, 128, 1) for name in names], axis=2)