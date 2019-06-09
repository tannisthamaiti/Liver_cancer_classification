#!/usr/bin/python

from __future__ import division

import numpy as np
import xgboost as xgb
import pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import KFold
from collections import Counter
from imblearn.over_sampling import SMOTE 
# This file attempts to do multilabel classification of spectrum samples based on xgboost
# and using SMOOTE method for class imbalance

#loading data and preprocessing

data_spectrum =  pd.read_csv('Spectrum_baseline.csv') #'Spectrum_baseline.csv'
labels = pd.read_csv('Spectrum_labels.csv').values
labels_new = pd.DataFrame({'Class':labels[:,1].astype('int64')})
total_df = pd.concat([data_spectrum, labels_new], axis =1)
feature = total_df.columns
data = total_df[feature[1:]].values
sz = data.shape
# shuffle data to get a good mix of classes
np.random.shuffle(data)
# test train split 70-30 
train = data[:int(sz[0] * 0.7), :]
test = data[int(sz[0] * 0.7):, :]
train_X = train[:, :1014]
train_Y = train[:, 1015]
test_X = test[:, :1014]
test_Y = test[:, 1015]
# taking into account imbalance class using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(train_X, train_Y)
xg_train = xgb.DMatrix(X_res, label=y_res)
xg_test = xgb.DMatrix(test_X, label=test_Y)
watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 5
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['scale_pos_weight']:[0.33, 0.7, 0.167,0.3,0.4]
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 5
# train xgboost
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))
# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 5)
pred_label = np.argmax(pred_prob, axis=1)
error_rate = np.sum(pred_label != test_Y) / test_Y.shape[0]
print('Test error using softprob = {}'.format(error_rate))
# Generate confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pred, test_Y)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)
print(recall, precision)
