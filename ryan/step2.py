#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:01:06 2018

@author: HSIN
"""

import time
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import metrics

stime = time.time()

np.random.seed(46)



test_fids = []
f = open('test_fids.txt', 'r')
for line in f:
    fid = line.replace('\n','')
    test_fids.append(fid)
f.close()



x = []
y = []


x_test = []
y_test = []
fid_test = []

f = open('output.csv')
for line in csv.reader(f):
#    print(line)
    
    
    if line[0] in test_fids:
        x_test.append([float(x) for x in line[1:-1]])
        fid_test.append(line[0])
    
    if line[-1] not in ['0', '1']:
        continue
    
    x.append([float(x) for x in line[1:-1]])
    y.append(int(line[-1]))
    
f.close()

x = np.asarray(x)
y = np.asarray(y)




train_valid_ratio = 0.9
indices = np.random.permutation(x.shape[0])
train_idx, valid_idx = indices[:int(x.shape[0] * train_valid_ratio)], indices[int(x.shape[0] * train_valid_ratio):]
x_train, x_valid = x[train_idx,:], x[valid_idx,:]
y_train, y_valid = y[train_idx], y[valid_idx]



rf = RandomForestClassifier(n_estimators = 200)

rf.fit(x_train, y_train)

train_pred = rf.predict_proba(x_train)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_train, train_pred, pos_label=1)
train_auc = metrics.auc(fpr, tpr)
print('RF Train AUC:', train_auc)

valid_pred = rf.predict_proba(x_valid)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_valid, valid_pred, pos_label=1)
valid_auc = metrics.auc(fpr, tpr)
print('RF Valid AUC:', valid_auc)

rf_train_pred = train_pred
rf_valid_pred = valid_pred



ext = ExtraTreesClassifier(n_estimators = 200)

ext.fit(x_train, y_train)

train_pred = ext.predict_proba(x_train)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_train, train_pred, pos_label=1)
train_auc = metrics.auc(fpr, tpr)
print('EXT Train AUC:', train_auc)

valid_pred = ext.predict_proba(x_valid)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_valid, valid_pred, pos_label=1)
valid_auc = metrics.auc(fpr, tpr)
print('EXT Valid AUC:', valid_auc)

ext_train_pred = train_pred
ext_valid_pred = valid_pred


xgb = XGBClassifier(max_depth = 20,
                    learning_rate = 0.01,
                    n_estimators = 200)

xgb.fit(x_train, y_train)

train_pred = xgb.predict_proba(x_train)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_train, train_pred, pos_label=1)
train_auc = metrics.auc(fpr, tpr)
print('XGB Train AUC:', train_auc)

valid_pred = xgb.predict_proba(x_valid)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_valid, valid_pred, pos_label=1)
valid_auc = metrics.auc(fpr, tpr)
print('XGB Valid  AUC:', valid_auc)


xgb_train_pred = train_pred
xgb_valid_pred = valid_pred




ensemble_train_pred = (rf_train_pred + ext_train_pred + xgb_train_pred)/3

fpr, tpr, thresholds = metrics.roc_curve(y_train, ensemble_train_pred, pos_label=1)
valid_auc = metrics.auc(fpr, tpr)
print('Ensemble Train AUC:', train_auc)


ensemble_valid_pred = (rf_valid_pred + ext_valid_pred + xgb_valid_pred)/3

fpr, tpr, thresholds = metrics.roc_curve(y_valid, ensemble_valid_pred, pos_label=1)
valid_auc = metrics.auc(fpr, tpr)
print('Ensemble Valid AUC:', valid_auc)


rf = RandomForestClassifier(n_estimators = 200)
rf.fit(x, y)
rf_test_pred = rf.predict_proba(x_test)[:,1]


ext = ExtraTreesClassifier(n_estimators = 200)
ext.fit(x, y)
ext_test_pred = ext.predict_proba(x_test)[:,1]


xgb = XGBClassifier(max_depth = 20,
                    learning_rate = 0.01,
                    n_estimators = 200)
xgb.fit(x, y)
xgb_test_pred = xgb.predict_proba(x_test)[:,1]


test_pred = (rf_test_pred + xgb_test_pred + ext_test_pred)/3

submit = []
for i in range(len(test_pred)):
    submit.append([fid_test[i], test_pred[i]])

f = open('submit.csv', 'w')
w = csv.writer(f)
w.writerows(submit)
f.close()


print('Time Taken:', time.time()-stime)
