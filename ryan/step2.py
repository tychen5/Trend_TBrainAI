#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:01:06 2018

@author: HSIN
"""
import sys
import time
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn import svm

stime = time.time()

np.random.seed(4646)
eps = 10 ** -5
estimators_num = 400




#############################################################################



test_fids = []
f = open('test_fids.txt', 'r')
for line in f:
    fid = line.replace('\n','')
    test_fids.append(fid)
f.close()



x = []
y = []


x_test = []
fid_test = []

title = []

f = open('output.csv')
for line in csv.reader(f):
#    print(line)
    
    if line[0] == 'fid':
        title = line[1:-2]
    
    if line[0] in test_fids:
        x_test.append([float(x) for x in line[1:-2]])
        fid_test.append(line[0])
    
    if line[-2] not in ['0', '1']:
        continue
    
    x.append([float(x) for x in line[1:-2]])
    y.append(int(line[-2]))
    
f.close()

x = np.asarray(x)
y = np.asarray(y)

x_test = np.asarray(x_test)





train_valid_ratio = 0.9
indices = np.random.permutation(x.shape[0])
train_idx, valid_idx = indices[:int(x.shape[0] * train_valid_ratio)], indices[int(x.shape[0] * train_valid_ratio):]
x_train, x_valid = x[train_idx,:], x[valid_idx,:]
y_train, y_valid = y[train_idx], y[valid_idx]


x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0) + eps

x_train = x_train - np.tile(x_train_mean,(len(x_train),1))
x_train = x_train/np.tile(x_train_std,(len(x_train),1))


x_valid = x_valid - np.tile(x_train_mean,(len(x_valid),1))
x_valid = x_valid/np.tile(x_train_std,(len(x_valid),1))


#train_pred = np.zeros(len(y_train))
#fpr, tpr, thresholds = metrics.roc_curve(y_train, train_pred, pos_label=1)
#train_auc = metrics.auc(fpr, tpr)
#print('ALL-ZERO Train AUC:', train_auc)
#
#valid_pred = np.zeros(len(y_valid))
#fpr, tpr, thresholds = metrics.roc_curve(y_valid, valid_pred, pos_label=1)
#valid_auc = metrics.auc(fpr, tpr)
#print('ALL-ZERO Valid AUC:', valid_auc)



print('Model Building...')

rf = RandomForestClassifier(n_estimators = estimators_num,
                            max_features = 0.5,
                            n_jobs = 4)

ext = ExtraTreesClassifier(n_estimators = estimators_num,
                           max_features = 0.5,
                           n_jobs = 4)

xgb = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.8,
                    colsample_bytree = 0.8,
                    max_depth = 15,
                    learning_rate = 0.05,
                    subsample = 0.9,
                    nthread = 4
                    )
                    



clfs = [rf, ext, xgb]
clf_names = ['RF', 'EXT', 'XGB']

train_preds = []
valid_preds = []

for i in range(len(clfs)):
    
    clf = clfs[i]
    clf_name = clf_names[i]
    
    clf.fit(x_train, y_train)
    
    train_pred = clf.predict_proba(x_train)[:,1]

    fpr, tpr, thresholds = metrics.roc_curve(y_train, train_pred, pos_label=1)
    train_auc = metrics.auc(fpr, tpr)
    print(clf_name, 'Train AUC:', train_auc)
    
    valid_pred = clf.predict_proba(x_valid)[:,1]
    
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, valid_pred, pos_label=1)
    valid_auc = metrics.auc(fpr, tpr)
    print(clf_name, 'Valid AUC:', valid_auc)
    
    train_preds.append(train_pred)
    valid_preds.append(valid_pred)
    
    print('Time Taken:', time.time()-stime)



ensemble_train_pred = (train_preds[0] + train_preds[1] + train_preds[2])/3

fpr, tpr, thresholds = metrics.roc_curve(y_train, ensemble_train_pred, pos_label=1)
train_auc = metrics.auc(fpr, tpr)
print('Ensemble Train AUC:', train_auc)


ensemble_valid_pred = (valid_preds[0] + valid_preds[1] + valid_preds[2])/3


fpr, tpr, thresholds = metrics.roc_curve(y_valid, ensemble_valid_pred, pos_label=1)
valid_auc = metrics.auc(fpr, tpr)
print('Ensemble Valid AUC:', valid_auc)






#############################################################################



test_fids = []
f = open('test_fids.txt', 'r')
for line in f:
    fid = line.replace('\n','')
    test_fids.append(fid)
f.close()



x = []
y = []


x_test = []
fid_test = []

title = []

f = open('output.csv')
for line in csv.reader(f):
#    print(line)
    
    if line[0] == 'fid':
        title = line[1:-2]
    
    if line[0] in test_fids:
        x_test.append([float(x) for x in line[1:-2]])
        fid_test.append(line[0])
    
    if line[-2] not in ['0', '1']:
        continue
    
    x.append([float(x) for x in line[1:-2]])
    y.append(int(line[-2]))
    
f.close()

x = np.asarray(x)
y = np.asarray(y)

x_test = np.asarray(x_test)


x_train = x
y_train = y
x_test = x_test

x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0) + eps


x_train = x_train - np.tile(x_train_mean,(len(x_train),1))
x_train = x_train/np.tile(x_train_std,(len(x_train),1))


x_test = x_test - np.tile(x_train_mean,(len(x_test),1))
x_test = x_test/np.tile(x_train_std,(len(x_test),1))



print('Model Building...')

rf = RandomForestClassifier(n_estimators = estimators_num,
                            max_features = 0.5,
                            n_jobs = 4)

ext = ExtraTreesClassifier(n_estimators = estimators_num,
                           max_features = 0.5,
                           n_jobs = 4)

xgb = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.8,
                    colsample_bytree = 0.8,
                    max_depth = 15,
                    learning_rate = 0.05,
                    subsample = 0.9,
                    nthread = 4)
                    

                    



#clfs = [rf, ext, xgb]
#clf_names = ['RF', 'EXT', 'XGB']

clfs = [xgb]
clf_names = ['XGB']

train_preds = []
test_preds = []

for i in range(len(clfs)):
    
    clf = clfs[i]
    clf_name = clf_names[i]
    
    clf.fit(x_train, y_train)
    
    train_pred = clf.predict_proba(x_train)[:,1]

    fpr, tpr, thresholds = metrics.roc_curve(y_train, train_pred, pos_label=1)
    train_auc = metrics.auc(fpr, tpr)
    print(clf_name, 'Train AUC:', train_auc)
    
    test_pred = clf.predict_proba(x_test)[:,1]
    
    train_preds.append(train_pred)
    test_preds.append(test_pred)
    
    print('Time Taken:', time.time()-stime)

#ensemble_test_pred = (test_preds[0] + test_preds[1] + test_preds[2])/3
    
ensemble_test_pred = test_preds[0]

submit = []
for i in range(len(ensemble_test_pred)):
    submit.append([fid_test[i], ensemble_test_pred[i]])

f = open('submit.csv', 'w')
w = csv.writer(f)
w.writerows(submit)
f.close()


print('Time Taken:', time.time()-stime)
