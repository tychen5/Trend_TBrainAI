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
eps = 10 ** -8
clf_num = 10
estimators_num = 600



fid2new_pid_infec_ratio = {}

count = 0
f = open('june_product_ratio_mean_new.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2new_pid_infec_ratio[line[0]] = float(line[1])
f.close()


mf_pid_10_title = []
for i in range(1,11):
    mf_pid_10_title.append('mf_pid_10_' + str(i))

fid2mf_pid_10 = {}
count = 0
f = open('frank_MF_pid10.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2mf_pid_10[line[0]] = [float(x) for x in line[1:]]
f.close()


mf_pid_5_title = []
for i in range(1,6):
    mf_pid_5_title.append('mf_pid_5_' + str(i))

fid2mf_pid_5 = {}
count = 0
f = open('frank_MF_pid5.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2mf_pid_5[line[0]] = [float(x) for x in line[1:]]
f.close()


mf_cid_50_title = []
for i in range(1,51):
    mf_cid_50_title.append('mf_cid_50_' + str(i))
    
fid2mf_cid_50 = {}
count = 0
f = open('frank_MF_cid50.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2mf_cid_50[line[0]] = [float(x) for x in line[1:]]
f.close()



mf_cid_30_title = []
for i in range(1,31):
    mf_cid_30_title.append('mf_cid_30_' + str(i))
    
fid2mf_cid_30 = {}
count = 0
f = open('frank_MF_cid30.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2mf_cid_30[line[0]] = [float(x) for x in line[1:]]
f.close()



mf_cid_15_title = []
for i in range(1,16):
    mf_cid_15_title.append('mf_cid_15_' + str(i))

fid2mf_cid_15 = {}
count = 0
f = open('frank_MF_cid15.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2mf_cid_15[line[0]] = [float(x) for x in line[1:]]
f.close()


fid2duration_long = {}
fid2discrete_rate = {}
count = 0
f = open('leo_duration_discreteRate.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2duration_long[line[1]] = float(line[2])
    fid2discrete_rate[line[1]] = float(line[3])
f.close()



duration_title = ['duration_q1','duration_q2','duration_q3','duration_std']
fid2duration_data = {}
count = 0
f = open('leo_duration_statistic1.csv')
for line in csv.reader(f):
    count += 1
    if count in [1,2,3]:
        continue
    fid2duration_data[line[0]] = [float(x) for x in line[1:5]]
f.close()



appear_hour_title = []
fid2appear_hour_data = {}
count = 0
f = open('ryan_time_feature.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        appear_hour_title = line[59:]
        sorting_idx = sorted(range(len(appear_hour_title)), key=lambda k: appear_hour_title[k])
        appear_hour_title = list(np.asarray(appear_hour_title)[sorting_idx])
        continue
    appear_hour_data = [float(x) for x in line[59:]]
    appear_hour_data = list(np.asarray(appear_hour_data)[sorting_idx])
    fid2appear_hour_data[line[0]] = appear_hour_data
    
f.close()

hour_title = []
fid2hour_data = {}
count = 0
f = open('june_hours.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        hour_title = line[2:]
        continue
    fid2hour_data[line[1]] = [float(x) for x in line[2:]]
f.close()




fid2h_infec_ratio = {}
count = 0
f = open('june_hour_infec.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2h_infec_ratio[line[1]] = float(line[2])
f.close()


fid2pid_infec_ratio = {}
count = 0
f = open('june_product_ratio.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2pid_infec_ratio[line[1]] = float(line[2])
f.close()



#fid2cid_infec_ratio = {}
#count = 0
#f = open('leo_cid_padCid_padFid.csv')
#for line in csv.reader(f):
#    count += 1
#    if count == 1:
#        continue
#    
#    ratio = float(line[2])
#    
#    if ratio >= 0 and ratio <= 0.25:
#        round_ratio = 1
#    elif ratio > 0.25 and ratio <= 0.50:
#        round_ratio = 2
#    elif ratio > 0.50 and ratio <= 0.75:
#        round_ratio = 3
#    elif ratio > 0.75 and ratio <= 1:
#        round_ratio = 4
#    
#    
#    fid2cid_infec_ratio[line[1]] = round_ratio
#    
#f.close()







#sys.exit()


#############################################################################

"""

x = []
y = []


x_test = []
fid_test = []
fid_train = []

title = []
the_selected_title = []

f = open('output.csv')
for line in csv.reader(f):
#    print(line)
    
    if line[0] == 'fid':
        
        title = line[1:-2]
        for t in title:
            
            if True and \
            not t.startswith('date') and \
            not t.startswith('pid_0374c4') and \
            not t.startswith('pid_05b409') and \
            not t.startswith('pid_3c2be6') and \
            not t.startswith('pid_aaa9c8') and \
            not t.startswith('pid_cc3a6a') and \
            not t.startswith('pid_8b7f69') and \
            not t.startswith('appear_day_7') and \
            not t.startswith('appear_day_6') and \
            not t.startswith('appear_day_5') and \
            not t.startswith('appear_day_4') and \
            not t.startswith('march') and \
            not t.startswith('april') and \
            not t.startswith('may') and \
            not (t.startswith('appear_') and t.endswith('_count'))  and \
            not (t.startswith('appear_') and t.endswith('d/p_ratio'))and \
            not (t.startswith('H_') and t.endswith('_count')) and \
            not (t.startswith('pid_') and t.endswith('_count')) and \
            not t in ['pid_0cdb7a_ratio', 'pid_218578_ratio', 'pid_75f310_ratio', 'pid_8452da_ratio', 'pid_fec24f_ratio',
                      'pid_262880_ratio', 'pid_a310bb_ratio', 'pid_26a5d0_ratio', 'pid_dd8d4a_ratio', 'pid_8541a0_ratio']:
                
                the_selected_title.append(t)
        
        
        for t in hour_title:
            
            if True and \
            not t.startswith('H_0_11_') and \
            not t.startswith('H_12_23_') and \
            not t.startswith('H_0_7_') and \
            not t.startswith('H_8_15_') and \
            not t.startswith('H_16_23_') and \
            not t.startswith('H_0_5_') and \
            not t.startswith('H_6_11_') and \
            not t.startswith('H_12_17_') and \
            not t.startswith('H_18_23_') and \
            not t.startswith('X12_') and \
            not t.startswith('X8_') and \
            not t.startswith('X6_') and \
            not t.startswith('H_') and \
            not t.endswith('_cnt'):
            
                the_selected_title.append(t)
            
#            the_selected_title.append(t)
        
        
        
        for t in appear_hour_title:
            
            if True and \
            not t.endswith('cnt') and \
            not t.endswith('increase_rate') and \
            not t.startswith('D1_H7') and \
            not t.startswith('D1_H6') and \
            not t.startswith('D1_H5') and \
            not t.startswith('D1_H4'):
                the_selected_title.append(t)
        
        
        
        for t in duration_title:
            
            if True:
                the_selected_title.append(t)
        
        
        
        for t in mf_cid_50_title:
            
            if False and \
            t not in ['mf_cid_50_10','mf_cid_50_11','mf_cid_50_12','mf_cid_50_13','mf_cid_50_18', \
                      'mf_cid_50_20','mf_cid_50_23','mf_cid_50_24','mf_cid_50_25','mf_cid_50_28', \
                      'mf_cid_50_29','mf_cid_50_33','mf_cid_50_35','mf_cid_50_39','mf_cid_50_44', \
                      'mf_cid_50_6','mf_cid_50_9']:
                the_selected_title.append(t)
        
        
        for t in mf_cid_30_title:
            
            if True and \
            t not in ['mf_cid_30_10','mf_cid_30_11','mf_cid_30_12','mf_cid_30_13','mf_cid_30_15', \
                      'mf_cid_30_18','mf_cid_30_19','mf_cid_30_20','mf_cid_30_23','mf_cid_30_24', \
                      'mf_cid_30_25','mf_cid_30_29','mf_cid_30_6']:
                the_selected_title.append(t)
                
        
        for t in mf_cid_15_title:
            
            if False and \
            t not in ['mf_cid_15_10','mf_cid_15_11','mf_cid_15_12','mf_cid_15_13', 'mf_cid_15_6']:
                the_selected_title.append(t)
        
        
        
        for t in mf_pid_10_title:
            
            if True:
                the_selected_title.append(t)
        
        for t in mf_pid_5_title:
            
            if False:
                the_selected_title.append(t)
                
                
                
        
        
        the_selected_title.append('h_infec_ratio')
        the_selected_title.append('pid_infec_ratio')
#        the_selected_title.append('cid_infec_ratio')
        the_selected_title.append('duration_long')
        the_selected_title.append('discrete_rate')
        
        the_selected_title.append('new_pid_infec_ratio')
        
        
        print(the_selected_title)
        continue
                
    
    
    the_fid = line[0]
    the_data = line[1:-2]
    the_selected_data = []
    
    for i in range(len(the_data)):
        if title[i] in the_selected_title:
            if 'count' in title[i]:
                the_selected_data.append(np.log(float(the_data[i]) + eps))
            else:
                the_selected_data.append(float(the_data[i]))
    
    
    
    the_hour_data = fid2hour_data[the_fid]
    for i in range(len(the_hour_data)):
        if hour_title[i] in the_selected_title:
            the_selected_data.append(float(the_hour_data[i]))
    
    
    the_appear_hour_data = fid2appear_hour_data[the_fid]
    for i in range(len(the_appear_hour_data)):
        if appear_hour_title[i] in the_selected_title:
            the_selected_data.append(float(the_appear_hour_data[i]))
    
        
    the_duration_data = fid2duration_data[the_fid]
    for i in range(len(the_duration_data)):
        if duration_title[i] in the_selected_title:
            the_selected_data.append(float(the_duration_data[i]))
    
    the_mf_cid_50_data = fid2mf_cid_50[the_fid]
    for i in range(len(the_mf_cid_50_data)):
        if mf_cid_50_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_cid_50_data[i]))
    
    the_mf_cid_30_data = fid2mf_cid_30[the_fid]
    for i in range(len(the_mf_cid_30_data)):
        if mf_cid_30_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_cid_30_data[i]))
    
    
    the_mf_cid_15_data = fid2mf_cid_15[the_fid]
    for i in range(len(the_mf_cid_15_data)):
        if mf_cid_15_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_cid_15_data[i]))
    
    
    the_mf_pid_10_data = fid2mf_pid_10[the_fid]
    for i in range(len(the_mf_pid_10_data)):
        if mf_pid_10_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_pid_10_data[i]))
    
    
    the_mf_pid_5_data = fid2mf_pid_5[the_fid]
    for i in range(len(the_mf_pid_5_data)):
        if mf_pid_5_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_pid_5_data[i]))
            
    
    the_selected_data.append(fid2h_infec_ratio[the_fid])
    the_selected_data.append(fid2pid_infec_ratio[the_fid])
#    the_selected_data.append(fid2cid_infec_ratio[the_fid])
    the_selected_data.append(fid2duration_long[the_fid])
    the_selected_data.append(fid2discrete_rate[the_fid])
    
    
    the_selected_data.append(fid2new_pid_infec_ratio[the_fid])
    
    
    if line[-1] == 'train':
        
        x.append(the_selected_data)
        y.append(int(line[-2]))
        fid_train.append(line[0])
    
    elif line[-1] == 'test':
        x_test.append(the_selected_data)
        fid_test.append(line[0])
    
    
f.close()




x = np.asarray(x)
y = np.asarray(y)
x_test = np.asarray(x_test)

print('x shape', x.shape)
print('y shape', y.shape)
print('x_test shape', x_test.shape)

#####################################################

train_selected_features = []
train_answers = []

train_selected_features.append(['fid'] + the_selected_title)
train_answers.append(['fid', 'ans'])

for i in range(x.shape[0]):
    feat = [fid_train[i]] + list(x[i,:])
    train_selected_features.append(feat)
    answer = [fid_train[i]] + [y[i]]
    train_answers.append(answer)

f = open('train_selected_features.csv', 'w')
w = csv.writer(f)
w.writerows(train_selected_features)
f.close()

f = open('train_answers.csv', 'w')
w = csv.writer(f)
w.writerows(train_answers)
f.close()



test_selected_features = []

test_selected_features.append(['fid'] + the_selected_title)

for i in range(x_test.shape[0]):
    feat = [fid_test[i]] + list(x_test[i,:])
    test_selected_features.append(feat)

f = open('test_selected_features.csv', 'w')
w = csv.writer(f)
w.writerows(test_selected_features)
f.close()

#sys.exit()

train_valid_ratio = 0.9
indices = np.random.permutation(x.shape[0])
train_idx, valid_idx = indices[:int(x.shape[0] * train_valid_ratio)], indices[int(x.shape[0] * train_valid_ratio):]
x_train, x_valid = x[train_idx,:], x[valid_idx,:]
y_train, y_valid = y[train_idx], y[valid_idx]


#x_train_mean = np.mean(x_train, axis=0)
#x_train_std = np.std(x_train, axis=0) + eps
#
#x_train = x_train - np.tile(x_train_mean,(len(x_train),1))
#x_train = x_train/np.tile(x_train_std,(len(x_train),1))
#
#
#x_valid = x_valid - np.tile(x_train_mean,(len(x_valid),1))
#x_valid = x_valid/np.tile(x_train_std,(len(x_valid),1))



x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0) + eps

x_train = x_train - np.tile(x_mean,(len(x_train),1))
x_train = x_train/np.tile(x_std,(len(x_train),1))


x_valid = x_valid - np.tile(x_mean,(len(x_valid),1))
x_valid = x_valid/np.tile(x_std,(len(x_valid),1))


#####################################################




print('Model Building...')

#rf = RandomForestClassifier(n_estimators = estimators_num,
#                            max_features = 0.5,
#                            n_jobs = 4)
#
#ext = ExtraTreesClassifier(n_estimators = estimators_num,
#                           max_features = 0.5,
#                           n_jobs = 4)

xgb1 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )


xgb2 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb3 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )


xgb4 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb5 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb6 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb7 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb8 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb9 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb10 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )


clfs = [xgb1, xgb2, xgb3, xgb4, xgb5, xgb6, xgb7, xgb8, xgb9, xgb10][:clf_num]
clf_names = ['XGB1', 'XGB2', 'XGB3', 'XGB4', 'XGB5', 'XGB6', 'XGB7', 'XGB8', 'XGB9', 'XGB10'][:clf_num]

#clfs = [xgb1, xgb2, xgb3]
#clf_names = ['XGB1', 'XGB2', 'XGB3']

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
    print('\n')



ensemble_train_pred = None
for train_pred in train_preds:
    if ensemble_train_pred is None:
        ensemble_train_pred = train_pred
    else:
        ensemble_train_pred = ensemble_train_pred * train_pred
ensemble_train_pred = ensemble_train_pred/np.max(ensemble_train_pred)

fpr, tpr, thresholds = metrics.roc_curve(y_train, ensemble_train_pred, pos_label=1)
train_auc = metrics.auc(fpr, tpr)
print('Ensemble Train AUC:', train_auc)



ensemble_valid_pred = None
for valid_pred in valid_preds:
    if ensemble_valid_pred is None:
        ensemble_valid_pred = valid_pred
    else:
        ensemble_valid_pred = ensemble_valid_pred * valid_pred
ensemble_valid_pred = ensemble_valid_pred/np.max(ensemble_valid_pred)

fpr, tpr, thresholds = metrics.roc_curve(y_valid, ensemble_valid_pred, pos_label=1)
valid_auc = metrics.auc(fpr, tpr)
print('Ensemble1 Valid AUC:', valid_auc)


ensemble_valid_pred = np.vstack(valid_preds).mean(axis=0)
ensemble_valid_pred = ensemble_valid_pred/np.max(ensemble_valid_pred)

fpr, tpr, thresholds = metrics.roc_curve(y_valid, ensemble_valid_pred, pos_label=1)
valid_auc = metrics.auc(fpr, tpr)
print('Ensemble2 Valid AUC:', valid_auc)


ensemble_valid_pred = np.vstack(valid_preds).max(axis=0)
ensemble_valid_pred = ensemble_valid_pred/np.max(ensemble_valid_pred)

fpr, tpr, thresholds = metrics.roc_curve(y_valid, ensemble_valid_pred, pos_label=1)
valid_auc = metrics.auc(fpr, tpr)
print('Ensemble3 Valid AUC:', valid_auc)


ensemble_valid_pred = np.vstack(valid_preds).min(axis=0)
ensemble_valid_pred = ensemble_valid_pred/np.max(ensemble_valid_pred)

fpr, tpr, thresholds = metrics.roc_curve(y_valid, ensemble_valid_pred, pos_label=1)
valid_auc = metrics.auc(fpr, tpr)
print('Ensemble4 Valid AUC:', valid_auc)

#sys.exit()


"""
#############################################################################



x = []
y = []


x_test = []
fid_test = []
fid_train = []

title = []
the_selected_title = []

f = open('output.csv')
for line in csv.reader(f):
#    print(line)
    
    if line[0] == 'fid':
        
        title = line[1:-2]
        for t in title:
            
            if True and \
            not t.startswith('date') and \
            not t.startswith('pid_0374c4') and \
            not t.startswith('pid_05b409') and \
            not t.startswith('pid_3c2be6') and \
            not t.startswith('pid_aaa9c8') and \
            not t.startswith('pid_cc3a6a') and \
            not t.startswith('pid_8b7f69') and \
            not t.startswith('appear_day_7') and \
            not t.startswith('appear_day_6') and \
            not t.startswith('appear_day_5') and \
            not t.startswith('appear_day_4') and \
            not t.startswith('march') and \
            not t.startswith('april') and \
            not t.startswith('may') and \
            not (t.startswith('appear_') and t.endswith('_count'))  and \
            not (t.startswith('appear_') and t.endswith('d/p_ratio'))and \
            not (t.startswith('H_') and t.endswith('_count')) and \
            not (t.startswith('pid_') and t.endswith('_count')) and \
            not t in ['pid_0cdb7a_ratio', 'pid_218578_ratio', 'pid_75f310_ratio', 'pid_8452da_ratio', 'pid_fec24f_ratio',
                      'pid_262880_ratio', 'pid_a310bb_ratio', 'pid_26a5d0_ratio', 'pid_dd8d4a_ratio', 'pid_8541a0_ratio']:
                
                the_selected_title.append(t)
        
        
        for t in hour_title:
            
            if True and \
            not t.startswith('H_0_11_') and \
            not t.startswith('H_12_23_') and \
            not t.startswith('H_0_7_') and \
            not t.startswith('H_8_15_') and \
            not t.startswith('H_16_23_') and \
            not t.startswith('H_0_5_') and \
            not t.startswith('H_6_11_') and \
            not t.startswith('H_12_17_') and \
            not t.startswith('H_18_23_') and \
            not t.startswith('X12_') and \
            not t.startswith('X8_') and \
            not t.startswith('X6_') and \
            not t.startswith('H_') and \
            not t.endswith('_cnt'):
            
                the_selected_title.append(t)
            
#            the_selected_title.append(t)
        
        
        
        for t in appear_hour_title:
            
            if True and \
            not t.endswith('cnt') and \
            not t.endswith('increase_rate') and \
            not t.startswith('D1_H7') and \
            not t.startswith('D1_H6') and \
            not t.startswith('D1_H5') and \
            not t.startswith('D1_H4'):
                the_selected_title.append(t)
        
        
        
        for t in duration_title:
            
            if True:
                the_selected_title.append(t)
        
        
        
        for t in mf_cid_50_title:
            
            if False and \
            t not in ['mf_cid_50_10','mf_cid_50_11','mf_cid_50_12','mf_cid_50_13','mf_cid_50_18', \
                      'mf_cid_50_20','mf_cid_50_23','mf_cid_50_24','mf_cid_50_25','mf_cid_50_28', \
                      'mf_cid_50_29','mf_cid_50_33','mf_cid_50_35','mf_cid_50_39','mf_cid_50_44', \
                      'mf_cid_50_6','mf_cid_50_9']:
                the_selected_title.append(t)
        
        
        for t in mf_cid_30_title:
            
            if True and \
            t not in ['mf_cid_30_10','mf_cid_30_11','mf_cid_30_12','mf_cid_30_13','mf_cid_30_15', \
                      'mf_cid_30_18','mf_cid_30_19','mf_cid_30_20','mf_cid_30_23','mf_cid_30_24', \
                      'mf_cid_30_25','mf_cid_30_29','mf_cid_30_6']:
                the_selected_title.append(t)
                
        
        for t in mf_cid_15_title:
            
            if False and \
            t not in ['mf_cid_15_10','mf_cid_15_11','mf_cid_15_12','mf_cid_15_13', 'mf_cid_15_6']:
                the_selected_title.append(t)
        
        
        
        for t in mf_pid_10_title:
            
            if True:
                the_selected_title.append(t)
        
        for t in mf_pid_5_title:
            
            if False:
                the_selected_title.append(t)
                
                
                
        
        
        the_selected_title.append('h_infec_ratio')
        the_selected_title.append('pid_infec_ratio')
#        the_selected_title.append('cid_infec_ratio')
        the_selected_title.append('duration_long')
        the_selected_title.append('discrete_rate')
        
        the_selected_title.append('new_pid_infec_ratio')
        
        
        print(the_selected_title)
        continue
                
    
    
    the_fid = line[0]
    the_data = line[1:-2]
    the_selected_data = []
    
    for i in range(len(the_data)):
        if title[i] in the_selected_title:
            if 'count' in title[i]:
                the_selected_data.append(np.log(float(the_data[i]) + eps))
            else:
                the_selected_data.append(float(the_data[i]))
    
    
    
    the_hour_data = fid2hour_data[the_fid]
    for i in range(len(the_hour_data)):
        if hour_title[i] in the_selected_title:
            the_selected_data.append(float(the_hour_data[i]))
    
    
    the_appear_hour_data = fid2appear_hour_data[the_fid]
    for i in range(len(the_appear_hour_data)):
        if appear_hour_title[i] in the_selected_title:
            the_selected_data.append(float(the_appear_hour_data[i]))
    
        
    the_duration_data = fid2duration_data[the_fid]
    for i in range(len(the_duration_data)):
        if duration_title[i] in the_selected_title:
            the_selected_data.append(float(the_duration_data[i]))
    
    the_mf_cid_50_data = fid2mf_cid_50[the_fid]
    for i in range(len(the_mf_cid_50_data)):
        if mf_cid_50_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_cid_50_data[i]))
    
    the_mf_cid_30_data = fid2mf_cid_30[the_fid]
    for i in range(len(the_mf_cid_30_data)):
        if mf_cid_30_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_cid_30_data[i]))
    
    
    the_mf_cid_15_data = fid2mf_cid_15[the_fid]
    for i in range(len(the_mf_cid_15_data)):
        if mf_cid_15_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_cid_15_data[i]))
    
    
    the_mf_pid_10_data = fid2mf_pid_10[the_fid]
    for i in range(len(the_mf_pid_10_data)):
        if mf_pid_10_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_pid_10_data[i]))
    
    
    the_mf_pid_5_data = fid2mf_pid_5[the_fid]
    for i in range(len(the_mf_pid_5_data)):
        if mf_pid_5_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_pid_5_data[i]))
            
    
    the_selected_data.append(fid2h_infec_ratio[the_fid])
    the_selected_data.append(fid2pid_infec_ratio[the_fid])
#    the_selected_data.append(fid2cid_infec_ratio[the_fid])
    the_selected_data.append(fid2duration_long[the_fid])
    the_selected_data.append(fid2discrete_rate[the_fid])
    
    
    the_selected_data.append(fid2new_pid_infec_ratio[the_fid])
    
    
    if line[-1] == 'train':
        
        x.append(the_selected_data)
        y.append(int(line[-2]))
        fid_train.append(line[0])
    
    elif line[-1] == 'test':
        x_test.append(the_selected_data)
        fid_test.append(line[0])
    
    
f.close()




x = np.asarray(x)
y = np.asarray(y)

x_test = np.asarray(x_test)

print('x shape', x.shape)
print('y shape', y.shape)

print('x_test shape', x_test.shape)




x_train = x
y_train = y
x_test = x_test




#x_train_mean = np.mean(x_train, axis=0)
#x_train_std = np.std(x_train, axis=0) + eps
#
#
#x_train = x_train - np.tile(x_train_mean,(len(x_train),1))
#x_train = x_train/np.tile(x_train_std,(len(x_train),1))
#
#
#x_test = x_test - np.tile(x_train_mean,(len(x_test),1))
#x_test = x_test/np.tile(x_train_std,(len(x_test),1))


x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0) + eps

x_train = x_train - np.tile(x_mean,(len(x_train),1))
x_train = x_train/np.tile(x_std,(len(x_train),1))

x_test = x_test - np.tile(x_mean,(len(x_test),1))
x_test = x_test/np.tile(x_std,(len(x_test),1))




print('Model Building...')

#rf = RandomForestClassifier(n_estimators = estimators_num,
#                            max_features = 0.5,
#                            n_jobs = 4)
#
#ext = ExtraTreesClassifier(n_estimators = estimators_num,
#                           max_features = 0.5,
#                           n_jobs = 4)

xgb1 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb2 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb3 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb4 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb5 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb6 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb7 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb8 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb9 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )

xgb10 = XGBClassifier(n_estimators = estimators_num,
                    colsample_bylevel = 0.6 + 0.4 * np.random.random(),
                    colsample_bytree = 0.6 + 0.4 * np.random.random(),
                    max_depth = 15 + int(10 * np.random.random()),
                    learning_rate = 0.05,
                    subsample = 0.6 + 0.4 * np.random.random(),
                    n_jobs = 4
                    )


clfs = [xgb1, xgb2, xgb3, xgb4, xgb5, xgb6, xgb7, xgb8, xgb9, xgb10][:clf_num]
clf_names = ['XGB1', 'XGB2', 'XGB3', 'XGB4', 'XGB5', 'XGB6', 'XGB7', 'XGB8', 'XGB9', 'XGB10'][:clf_num]

#clfs = [xgb1, xgb2, xgb3, xgb4, xgb5]
#clf_names = ['XGB1', 'XGB2', 'XGB3', 'XGB4', 'XGB5']

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
    print('\n')




ensemble_train_pred = None
for train_pred in train_preds:
    if ensemble_train_pred is None:
        ensemble_train_pred = train_pred
    else:
        ensemble_train_pred = ensemble_train_pred * train_pred
ensemble_train_pred = ensemble_train_pred/np.max(ensemble_train_pred)

fpr, tpr, thresholds = metrics.roc_curve(y_train, ensemble_train_pred, pos_label=1)
train_auc = metrics.auc(fpr, tpr)
print('Ensemble Train AUC:', train_auc)


ensemble_test_pred = None
for test_pred in test_preds:
    if ensemble_test_pred is None:
        ensemble_test_pred = test_pred
    else:
        ensemble_test_pred = ensemble_test_pred * test_pred
ensemble_test_pred = ensemble_test_pred/np.max(ensemble_test_pred)


#ensemble_test_pred = np.vstack(test_preds).mean(axis=0)
#ensemble_test_pred = ensemble_test_pred/np.max(ensemble_test_pred)


#ensemble_test_pred = np.vstack(test_preds).max(axis=0)
#ensemble_test_pred = ensemble_test_pred/np.max(ensemble_test_pred)
#
#
#ensemble_test_pred = np.vstack(test_preds).min(axis=0)
#ensemble_test_pred = ensemble_test_pred/np.max(ensemble_test_pred)
    

submit = []
for i in range(len(ensemble_test_pred)):
    submit.append([fid_test[i], ensemble_test_pred[i]])

f = open('submit.csv', 'w')
w = csv.writer(f)
w.writerows(submit)
f.close()


print('Time Taken:', time.time()-stime)
