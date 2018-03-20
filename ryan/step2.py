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
from sklearn.metrics import confusion_matrix

stime = time.time()

np.random.seed(4646)
eps = 10 ** -8
best_ratio = 0.2
clf_num = 50
estimators_num = 800



appear_uni_stat_title = []
fid2appear_uni_stat_data = {}
count = 0
f = open('june_day_uni_statistic.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        appear_uni_stat_title = line[1:]
        continue
    fid2appear_uni_stat_data[line[0]] = [float(x) for x in line[1:]]
f.close()


appear_pid_uni_title = []
fid2appear_pid_uni_data = {}
count = 0
f = open('june_product_day1-7.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        appear_pid_uni_title = line[1:]
        continue
    fid2appear_pid_uni_data[line[0]] = [float(x) for x in line[1:]]
f.close()


appear_cid_uni_title = []
fid2appear_cid_uni_data = {}
count = 0
f = open('june_customer_day1-7.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        appear_cid_uni_title = line[1:]
        continue
    fid2appear_cid_uni_data[line[0]] = [float(x) for x in line[1:]]
f.close()


uni_title = ['uni_cus_cnt','uni_cus_ratio','uni_pro_cnt','uni_pro_ratio']
fid2uni_data = {}
count = 0
f = open('june_unique_counts_and_ratio.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2uni_data[line[0]] = [float(x) for x in line[1:]]
f.close()


fid2new_cid_infec_ratio = {}

count = 0
#f = open('june_customer_ratio_mean_new.csv')
f = open('june_cus_ratio_padding_twice.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2new_cid_infec_ratio[line[0]] = float(line[1])
f.close()


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




mf_pid_ratio_10_title = []
for i in range(1,11):
    mf_pid_ratio_10_title.append('mf_pid_ratio_10_' + str(i))

fid2mf_pid_ratio_10 = {}
count = 0
f = open('frank_MF_pid_ratio_10.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2mf_pid_ratio_10[line[0]] = [float(x) for x in line[1:]]
f.close()


mf_pid_ratio_15_title = []
for i in range(1,16):
    mf_pid_ratio_15_title.append('mf_pid_ratio_15_' + str(i))

fid2mf_pid_ratio_15 = {}
count = 0
f = open('frank_MF_pid_ratio_15.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2mf_pid_ratio_15[line[0]] = [float(x) for x in line[1:]]
f.close()



mf_cid_ratio_15_title = []
for i in range(1,16):
    mf_cid_ratio_15_title.append('mf_cid_ratio_15_' + str(i))

fid2mf_cid_ratio_15 = {}
count = 0
f = open('frank_MF_cid_ratio_15.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2mf_cid_ratio_15[line[0]] = [float(x) for x in line[1:]]
f.close()


mf_cid_ratio_30_title = []
for i in range(1,31):
    mf_cid_ratio_30_title.append('mf_cid_ratio_30_' + str(i))

fid2mf_cid_ratio_30 = {}
count = 0
f = open('frank_MF_cid_ratio_30.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2mf_cid_ratio_30[line[0]] = [float(x) for x in line[1:]]
f.close()



mf_cid_ratio_50_title = []
for i in range(1,51):
    mf_cid_ratio_50_title.append('mf_cid_ratio_50_' + str(i))

fid2mf_cid_ratio_50 = {}
count = 0
f = open('frank_MF_cid_ratio_50.csv')
for line in csv.reader(f):
    count += 1
    if count == 1:
        continue
    fid2mf_cid_ratio_50[line[0]] = [float(x) for x in line[1:]]
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

x = []
y = []


x_test = []
fid_test = []
fid_train = []

title = []
the_selected_title = []

take_all = False

f = open('output.csv')
for line in csv.reader(f):
#    print(line)
    
    if line[0] == 'fid':
        
        title = line[1:-2]
        for t in title:
            
            if take_all or \
            (True and \
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
                      'pid_262880_ratio', 'pid_a310bb_ratio', 'pid_26a5d0_ratio', 'pid_dd8d4a_ratio', 'pid_8541a0_ratio'] and \
            not t.endswith('match_count')
            ):
                the_selected_title.append(t)
        
        
        for t in hour_title:
            
            if take_all or \
            (True and \
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
            not t.endswith('_cnt')
            ):
            
                the_selected_title.append(t)
        
        
        
        for t in appear_hour_title:
            
            if take_all or \
            (True and \
            not t.endswith('cnt') and \
            not t.endswith('increase_rate') and \
            not t.startswith('D1_H7') and \
            not t.startswith('D1_H6') and \
            not t.startswith('D1_H5') and \
            not t.startswith('D1_H4')
            ):
                the_selected_title.append(t)
        
        
        
        for t in duration_title:
            
            if take_all or \
            (True):
                the_selected_title.append(t)
        
        
        for t in uni_title:
            
            if take_all or \
            (True):
                the_selected_title.append(t)
        
        
        for t in appear_pid_uni_title:
            
            if take_all or \
            (False and \
             not t.endswith('_cnt') and \
             not t.startswith('day7') and \
             not t.endswith('day_ratio')
             ):
                the_selected_title.append(t)
        
        
        for t in appear_cid_uni_title:
            
            if take_all or \
            (False and \
             not t.endswith('_cnt') and \
             not t.startswith('day7') and \
             not t.endswith('day_ratio')
             ):
                the_selected_title.append(t)
        
        
        for t in appear_uni_stat_title:
            
            if take_all or \
            (False and \
             'cnt' not in t and \
             'day' not in t and \
             'pro' not in t
             ):
                the_selected_title.append(t)
        
        
        
        for t in mf_cid_50_title:
            
            if take_all or \
            (False and \
            t not in ['mf_cid_50_10','mf_cid_50_11','mf_cid_50_12','mf_cid_50_13','mf_cid_50_18', \
                      'mf_cid_50_20','mf_cid_50_23','mf_cid_50_24','mf_cid_50_25','mf_cid_50_28', \
                      'mf_cid_50_29','mf_cid_50_33','mf_cid_50_35','mf_cid_50_39','mf_cid_50_44', \
                      'mf_cid_50_6','mf_cid_50_9']
            ):
                the_selected_title.append(t)
        
        
        for t in mf_cid_30_title:
            
            if take_all or \
            (True and \
            t not in ['mf_cid_30_10','mf_cid_30_11','mf_cid_30_12','mf_cid_30_13','mf_cid_30_15', \
                      'mf_cid_30_18','mf_cid_30_19','mf_cid_30_20','mf_cid_30_23','mf_cid_30_24', \
                      'mf_cid_30_25','mf_cid_30_29','mf_cid_30_6']
            ):
                the_selected_title.append(t)
                
        
        for t in mf_cid_15_title:
            
            if take_all or \
            (False and \
            t not in ['mf_cid_15_10','mf_cid_15_11','mf_cid_15_12','mf_cid_15_13', 'mf_cid_15_6']
            ):
                the_selected_title.append(t)
        
        
        
        for t in mf_pid_10_title:
            
            if take_all or \
            (True):
                the_selected_title.append(t)
        
        for t in mf_pid_5_title:
            
            if take_all or \
            (False):
                the_selected_title.append(t)
        
        
        
        for t in mf_cid_ratio_50_title:
            
            if take_all or \
            (False):
                the_selected_title.append(t)
        
        
        for t in mf_cid_ratio_30_title:
            
            if take_all or \
            (True):
                the_selected_title.append(t)
        
        
        for t in mf_cid_ratio_15_title:
            
            if take_all or \
            (False):
                the_selected_title.append(t)
                
        
        
        for t in mf_pid_ratio_15_title:
            
            if take_all or \
            (False):
                the_selected_title.append(t)
        
        
        
        for t in mf_pid_ratio_10_title:
            
            if take_all or \
            (True):
                the_selected_title.append(t)
        
                
                
        
        
        the_selected_title.append('h_infec_ratio')
        the_selected_title.append('pid_infec_ratio')
#        the_selected_title.append('cid_infec_ratio')
        the_selected_title.append('duration_long')
        the_selected_title.append('discrete_rate')
        
        the_selected_title.append('new_pid_infec_ratio')
#        the_selected_title.append('new_cid_infec_ratio')
        
        
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
    
    the_uni_data = fid2uni_data[the_fid]
    for i in range(len(the_uni_data)):
        if uni_title[i] in the_selected_title:
            the_selected_data.append(float(the_uni_data[i]))
    
    
    the_appear_pid_uni_data = fid2appear_pid_uni_data[the_fid]
    for i in range(len(the_appear_pid_uni_data)):
        if appear_pid_uni_title[i] in the_selected_title:
            the_selected_data.append(float(the_appear_pid_uni_data[i]))
    
    
    the_appear_cid_uni_data = fid2appear_cid_uni_data[the_fid]
    for i in range(len(the_appear_cid_uni_data)):
        if appear_cid_uni_title[i] in the_selected_title:
            the_selected_data.append(float(the_appear_cid_uni_data[i]))
    
    
    the_appear_uni_stat_data = fid2appear_uni_stat_data[the_fid]
    for i in range(len(the_appear_uni_stat_data)):
        if appear_uni_stat_title[i] in the_selected_title:
            the_selected_data.append(float(the_appear_uni_stat_data[i]))
    
    
    
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
    
    
    
    the_mf_cid_ratio_50_data = fid2mf_cid_ratio_50[the_fid]
    for i in range(len(the_mf_cid_ratio_50_data)):
        if mf_cid_ratio_50_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_cid_ratio_50_data[i]))
            
    
    the_mf_cid_ratio_30_data = fid2mf_cid_ratio_30[the_fid]
    for i in range(len(the_mf_cid_ratio_30_data)):
        if mf_cid_ratio_30_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_cid_ratio_30_data[i]))
    
    
    the_mf_cid_ratio_15_data = fid2mf_cid_ratio_15[the_fid]
    for i in range(len(the_mf_cid_ratio_15_data)):
        if mf_cid_ratio_15_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_cid_ratio_15_data[i]))
    
    
    the_mf_pid_ratio_15_data = fid2mf_pid_ratio_15[the_fid]
    for i in range(len(the_mf_pid_ratio_15_data)):
        if mf_pid_ratio_15_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_pid_ratio_15_data[i]))        
    
    
    the_mf_pid_ratio_10_data = fid2mf_pid_ratio_10[the_fid]
    for i in range(len(the_mf_pid_ratio_10_data)):
        if mf_pid_ratio_10_title[i] in the_selected_title:
            the_selected_data.append(float(the_mf_pid_ratio_10_data[i]))
            
    
    the_selected_data.append(fid2h_infec_ratio[the_fid])
    the_selected_data.append(fid2pid_infec_ratio[the_fid])
#    the_selected_data.append(fid2cid_infec_ratio[the_fid])
    the_selected_data.append(fid2duration_long[the_fid])
    the_selected_data.append(fid2discrete_rate[the_fid])
    
    
    the_selected_data.append(fid2new_pid_infec_ratio[the_fid])
#    the_selected_data.append(fid2new_cid_infec_ratio[the_fid])
    
    
    if line[-1] == 'train':
        
        x.append(the_selected_data)
        y.append(int(line[-2]))
        fid_train.append(line[0])
                
        
    elif line[-1] == 'test':
        x_test.append(the_selected_data)
        fid_test.append(line[0])
    
    
f.close()


fid_train = np.asarray(fid_train)
fid_test = np.asarray(fid_test)

x = np.asarray(x)
y = np.asarray(y)
x_test = np.asarray(x_test)

print('x shape', x.shape)
print('y shape', y.shape)
print('x_test shape', x_test.shape)

#############################################################################
print('Output Selected Features...')
#############################################################################

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


#############################################################################
print('\n')
print('<Validating>')
#############################################################################




train_valid_ratio = 0.9
indices = np.random.permutation(x.shape[0])
train_idx, valid_idx = indices[:int(x.shape[0] * train_valid_ratio)], indices[int(x.shape[0] * train_valid_ratio):]
x_train, x_valid = x[train_idx,:], x[valid_idx,:]
y_train, y_valid = y[train_idx], y[valid_idx]
fid_train_train, fid_train_valid = fid_train[train_idx], fid_train[valid_idx]


x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0) + eps

x_train = x_train - np.tile(x_mean,(len(x_train),1))
x_train = x_train/np.tile(x_std,(len(x_train),1))


x_valid = x_valid - np.tile(x_mean,(len(x_valid),1))
x_valid = x_valid/np.tile(x_std,(len(x_valid),1))


#x_train_add = []
#y_train_add = []
#for i in range(len(y_train)):
#    if y_train[i] == 1:
#        for j in range(9 - 1):
#            x_train_add.append(x_train[i,:])
#            y_train_add.append(y_train[i])
#
#x_train_add = np.asarray(x_train_add)
#y_train_add = np.asarray(y_train_add)         
#
#x_train = np.vstack([x_train, x_train_add])
#y_train = np.append(y_train, y_train_add)


print('x_train shape', x_train.shape)
print('y_train shape', y_train.shape)

#####################################################




print('Model Building...')


clfs = []
clf_names = []
clf_params = []

for i in range(clf_num):
    
    param = {
            'n_estimators': estimators_num,
            'colsample_bylevel': 0.6 + 0.4 * np.random.random(),
            'colsample_bytree': 0.6 + 0.4 * np.random.random(),
            'max_depth': 18 + int(10 * np.random.random()),
            'subsample': 0.6 + 0.4 * np.random.random(),
            'base_score': np.random.random(),
#            'min_child_weight': int(1 + 3 * np.random.random()),
#            'scale_pos_weight': 9,
            'learning_rate': 0.05,
            'n_jobs': 4
            }
    
    xgb = XGBClassifier(**param)
    
    clfs.append(xgb)
    clf_names.append('XGB' + str(i+1))
    clf_params.append(param)


print('Model Training...')


train_preds = []
valid_preds = []
valid_aucs = []

for i in range(len(clfs)):
    
    clf = clfs[i]
    clf_name = clf_names[i]
    
    print(clf)
    print('\n')
    
    clf.fit(x_train, y_train)
    
    train_pred = clf.predict_proba(x_train)[:,1]

    fpr, tpr, thresholds = metrics.roc_curve(y_train, train_pred, pos_label=1)
    train_auc = metrics.auc(fpr, tpr)
    print(clf_name, 'Train AUC:', train_auc)
    
    valid_pred = clf.predict_proba(x_valid)[:,1]
    
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, valid_pred, pos_label=1)
    valid_auc = metrics.auc(fpr, tpr)
    print(clf_name, 'Valid AUC:', valid_auc)
    valid_aucs.append(valid_auc)
    
    
    conf_mat = confusion_matrix(y_valid, np.round(valid_pred), labels = [1, 0])
    print(conf_mat)
    
#    for i in range(len(y_valid)):
#        if round(valid_pred[i]) != y_valid[i]:
#            print(fid_train_valid[i])
#            print('pred: ', round(valid_pred[i]))
#            print('true: ', round(y_valid[i]))
    
    
    train_preds.append(train_pred)
    valid_preds.append(valid_pred)
    
    print('Time Taken:', time.time()-stime)
    print('\n\n')
    

the_best_aucs = []
the_best_clfs = []
the_best_params = []
the_best_valid_preds = []

for i,j,k,l in sorted(zip(valid_aucs, clfs, clf_params, valid_preds), reverse = True):
    the_best_aucs.append(i)
    the_best_clfs.append(j)
    the_best_params.append(k)
    the_best_valid_preds.append(l)



#ensemble_train_pred = None
#for train_pred in train_preds:
#    if ensemble_train_pred is None:
#        ensemble_train_pred = train_pred
#    else:
#        ensemble_train_pred = ensemble_train_pred * train_pred
#ensemble_train_pred = ensemble_train_pred/np.max(ensemble_train_pred)
#
#fpr, tpr, thresholds = metrics.roc_curve(y_train, ensemble_train_pred, pos_label=1)
#train_auc = metrics.auc(fpr, tpr)
#print('Ensemble Train AUC:', train_auc)






valid_preds = the_best_valid_preds[:int(best_ratio * clf_num)]




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


#############################################################################
print('\n')
print('<Testing>')
#############################################################################


x = np.asarray(x)
y = np.asarray(y)

x_test = np.asarray(x_test)

print('x shape', x.shape)
print('y shape', y.shape)

print('x_test shape', x_test.shape)




x_train = x
y_train = y
x_test = x_test


x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0) + eps

x_train = x_train - np.tile(x_mean,(len(x_train),1))
x_train = x_train/np.tile(x_std,(len(x_train),1))

x_test = x_test - np.tile(x_mean,(len(x_test),1))
x_test = x_test/np.tile(x_std,(len(x_test),1))




#x_train_add = []
#y_train_add = []
#for i in range(len(y_train)):
#    if y_train[i] == 1:
#        for j in range(9 - 1):
#            x_train_add.append(x_train[i,:])
#            y_train_add.append(y_train[i])
#
#x_train_add = np.asarray(x_train_add)
#y_train_add = np.asarray(y_train_add)         
#
#x_train = np.vstack([x_train, x_train_add])
#y_train = np.append(y_train, y_train_add)


print('x_train shape', x_train.shape)
print('y_train shape', y_train.shape)

#####################################################




print('Model Building...')


clfs = []
clf_names = []
clf_params = the_best_params[:int(best_ratio * clf_num)]

#for i in range(clf_num):
#    
#    param = {
#            'n_estimators': estimators_num,
#            'colsample_bylevel': 0.6 + 0.4 * np.random.random(),
#            'colsample_bytree': 0.6 + 0.4 * np.random.random(),
#            'max_depth': 15 + int(10 * np.random.random()),
#            'subsample': 0.6 + 0.4 * np.random.random(),
#            'base_score': np.random.random(),
##            'min_child_weight': int(1 + 3 * np.random.random()),
##            'scale_pos_weight': 9,
#            'learning_rate': 0.05,
#            'n_jobs': 4
#            }
#    
#    xgb = XGBClassifier(**param)
#    
#    clfs.append(xgb)
#    clf_names.append('XGB' + str(i+1))


for i in range(len(clf_params)):
    
    param = clf_params[i]
    
    xgb = XGBClassifier(**param)
    
    clfs.append(xgb)
    clf_names.append('XGB' + str(i+1))




print('Model Training...')



train_preds = []
test_preds = []

for i in range(len(clfs)):
    
    clf = clfs[i]
    clf_name = clf_names[i]
    
    print(clf)
    print('\n')
    
    clf.fit(x_train, y_train)
    
    train_pred = clf.predict_proba(x_train)[:,1]

    fpr, tpr, thresholds = metrics.roc_curve(y_train, train_pred, pos_label=1)
    train_auc = metrics.auc(fpr, tpr)
    print(clf_name, 'Train AUC:', train_auc)
    
    test_pred = clf.predict_proba(x_test)[:,1]
    
    train_preds.append(train_pred)
    test_preds.append(test_pred)
    
    print('Time Taken:', time.time()-stime)
    print('\n\n')




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
