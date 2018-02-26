#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 00:23:02 2018

@author: HSIN
"""

import csv
import os
import sys
import datetime
import time
import pickle
import numpy as np


stime = time.time()

train_path = './training-set.csv'
test_path = './testing-set.csv'
exception_dir = './exception'
query_log_dir = './query_log'

query_files = os.listdir(query_log_dir)
query_files = sorted(query_files)

#fids = set()
#cids = set()
#pids = set()
#
#for date_file in query_files:
#    print(date_file)
#    f = open(os.path.join(query_log_dir, date_file), 'r')
#    for line in csv.reader(f):
#        
#        fid = line[0]
#        cid = line[1]
##        qts = int(line[2])
#        pid = line[3]
#        
#        fids.add(fid)
#        cids.add(cid)
#        pids.add(pid)
#
#
#f = open('fids.txt', 'w')
#for fid in sorted(fids):
#    f.write(fid + '\n')
#f.close()
#
#
#f = open('cids.txt', 'w')
#for cid in sorted(cids):
#    f.write(cid + '\n')
#f.close()
#
#
#f = open('pids.txt', 'w')
#for pid in sorted(pids):
#    f.write(pid + '\n')
#f.close()




fids = set()
cids = set()
pids = set()


f = open('fids.txt', 'r')
for line in f:
    fid = line.replace('\n','')
    fids.add(fid)
f.close()

f = open('cids.txt', 'r')
for line in f:
    cid = line.replace('\n','')
    cids.add(cid)
f.close()

f = open('pids.txt', 'r')
for line in f:
    pid = line.replace('\n','')
    pids.add(pid)
f.close()




fid2data = {}

for fid in fids:
    
    fid2data[fid] = {}
    
    fid2data[fid]['H_00_count'] = 0
    fid2data[fid]['H_01_count'] = 0
    fid2data[fid]['H_02_count'] = 0
    fid2data[fid]['H_03_count'] = 0
    fid2data[fid]['H_04_count'] = 0
    fid2data[fid]['H_05_count'] = 0
    fid2data[fid]['H_06_count'] = 0
    fid2data[fid]['H_07_count'] = 0
    fid2data[fid]['H_08_count'] = 0
    fid2data[fid]['H_09_count'] = 0
    fid2data[fid]['H_10_count'] = 0
    fid2data[fid]['H_11_count'] = 0
    fid2data[fid]['H_12_count'] = 0
    fid2data[fid]['H_13_count'] = 0
    fid2data[fid]['H_14_count'] = 0
    fid2data[fid]['H_15_count'] = 0
    fid2data[fid]['H_16_count'] = 0
    fid2data[fid]['H_17_count'] = 0
    fid2data[fid]['H_18_count'] = 0
    fid2data[fid]['H_19_count'] = 0
    fid2data[fid]['H_20_count'] = 0
    fid2data[fid]['H_21_count'] = 0
    fid2data[fid]['H_22_count'] = 0
    fid2data[fid]['H_23_count'] = 0
    
    
    
    
    for query_file in query_files:
        date = query_file[:4]
        fid2data[fid]['date_'+date+'_count'] = 0
        
    for pid in pids:
        fid2data[fid]['pid_'+pid+'_count'] = 0
        
        
#    for query_file in query_files:
#        date = query_file[:4]
#        for pid in pids:
#            fid2data[fid]['date-pid_'+date+'-'+pid+'_count'] = 0
    
    
    fid2data[fid]['qts_list'] = []
    fid2data[fid]['qts_step_list'] = []
    
    fid2data[fid]['qts_step_mean'] = 0
    fid2data[fid]['qts_step_std'] = 0
    fid2data[fid]['qts_step_min'] = 0
    fid2data[fid]['qts_step_Q1'] = 0
    fid2data[fid]['qts_step_Q2'] = 0
    fid2data[fid]['qts_step_Q3'] = 0
    fid2data[fid]['qts_step_max'] = 0
    
    fid2data[fid]['count'] = 0
    
    
    fid2data[fid]['y'] = None
    
    






train_excepts = []
f = open(os.path.join(exception_dir, 'exception_train.txt'), 'r')
for line in f:
    the_id = line.replace('\n','')
    train_excepts.append(the_id)
f.close()


test_excepts = []
f = open(os.path.join(exception_dir, 'exception_testing.txt'), 'r')
for line in f:
    the_id = line.replace('\n','')
    test_excepts.append(the_id)
f.close()



train_fids = set()
f = open(train_path, 'r')
for line in csv.reader(f):
    if line[0] not in train_excepts:
        train_fids.add(line[0])
        fid2data[line[0]]['y'] = int(line[1])
f.close()
train_fids = sorted(train_fids)


test_fids = set()
f = open(test_path, 'r')
for line in csv.reader(f):
    if line[0] not in test_excepts:
        test_fids.add(line[0])
f.close()
test_fids = sorted(test_fids)


f = open('train_fids.txt', 'w')
for fid in sorted(train_fids):
    f.write(fid + '\n')
f.close()


f = open('test_fids.txt', 'w')
for fid in sorted(test_fids):
    f.write(fid + '\n')
f.close()


#sys.exit()

for date_file in query_files:
    print(date_file)
    f = open(os.path.join(query_log_dir, date_file), 'r')
    
    for line in csv.reader(f):
        
        fid = line[0]
        cid = line[1]
        qts = int(line[2])
        pid = line[3]
        
        the_time = datetime.datetime.utcfromtimestamp(qts)
        
        the_time_H = the_time.strftime('%H')
        the_time_date = the_time.strftime('%m%d')
        
        fid2data[fid]['H_'+the_time_H+'_count'] += 1
        fid2data[fid]['date_'+the_time_date+'_count'] += 1
        
        fid2data[fid]['pid_'+pid+'_count'] += 1
        
#        fid2data[fid]['date-pid_'+date+'-'+pid+'_count'] += 1
        
        fid2data[fid]['qts_list'].append(qts)
        
        fid2data[fid]['count'] += 1
        
        
    f.close()





for fid in fids:
    fid2data[fid]['qts_list'] = sorted(fid2data[fid]['qts_list']) 
    for i in range(len(fid2data[fid]['qts_list'])):
        if i == 0:
            continue
        else:
            the_qts_step = fid2data[fid]['qts_list'][i] - fid2data[fid]['qts_list'][i - 1]
            if the_qts_step < 0:
                sys.exit()
            fid2data[fid]['qts_step_list'].append(the_qts_step)



for fid in fids:
    
    the_qts_step_list = fid2data[fid]['qts_step_list']
    
    if not the_qts_step_list:
        continue
    
    fid2data[fid]['qts_step_mean'] = np.mean(the_qts_step_list)
    fid2data[fid]['qts_step_std'] = np.std(the_qts_step_list)
    fid2data[fid]['qts_step_min'] = np.min(the_qts_step_list)
    fid2data[fid]['qts_step_Q1'] = np.percentile(the_qts_step_list, 25)
    fid2data[fid]['qts_step_Q2'] = np.percentile(the_qts_step_list, 50)
    fid2data[fid]['qts_step_Q3'] = np.percentile(the_qts_step_list, 75)
    fid2data[fid]['qts_step_max'] = np.max(the_qts_step_list)


dataset = []
data = []
data.append('fid')
data.append('H_00_count')
data.append('H_01_count')
data.append('H_02_count')
data.append('H_03_count')
data.append('H_04_count')
data.append('H_05_count')
data.append('H_06_count')
data.append('H_07_count')
data.append('H_08_count')
data.append('H_09_count')
data.append('H_10_count')
data.append('H_11_count')
data.append('H_12_count')
data.append('H_13_count')
data.append('H_14_count')
data.append('H_15_count')
data.append('H_16_count')
data.append('H_17_count')
data.append('H_18_count')
data.append('H_19_count')
data.append('H_20_count')
data.append('H_21_count')
data.append('H_22_count')
data.append('H_23_count')


for query_file in query_files:
    date = query_file[:4]
    data.append('date_'+date+'_count')
    

for pid in pids:
    data.append('pid_'+pid+'_count')


#for query_file in query_files:
#    date = query_file[:4]
#    for pid in pids:
#        data.append('date-pid_'+date+'-'+pid+'_count')


data.append('qts_step_mean')
data.append('qts_step_std')
data.append('qts_step_min')
data.append('qts_step_Q1')
data.append('qts_step_Q2')
data.append('qts_step_Q3')
data.append('qts_step_max')

data.append('count')

data.append('y')

dataset.append(data)






for fid in fid2data:
    data = []
    
    data.append(fid)
    
    data.append(fid2data[fid]['H_00_count'])
    data.append(fid2data[fid]['H_01_count'])
    data.append(fid2data[fid]['H_02_count'])
    data.append(fid2data[fid]['H_03_count'])
    data.append(fid2data[fid]['H_04_count'])
    data.append(fid2data[fid]['H_05_count'])
    data.append(fid2data[fid]['H_06_count'])
    data.append(fid2data[fid]['H_07_count'])
    data.append(fid2data[fid]['H_08_count'])
    data.append(fid2data[fid]['H_09_count'])
    data.append(fid2data[fid]['H_10_count'])
    data.append(fid2data[fid]['H_11_count'])
    data.append(fid2data[fid]['H_12_count'])
    data.append(fid2data[fid]['H_13_count'])
    data.append(fid2data[fid]['H_14_count'])
    data.append(fid2data[fid]['H_15_count'])
    data.append(fid2data[fid]['H_16_count'])
    data.append(fid2data[fid]['H_17_count'])
    data.append(fid2data[fid]['H_18_count'])
    data.append(fid2data[fid]['H_19_count'])
    data.append(fid2data[fid]['H_20_count'])
    data.append(fid2data[fid]['H_21_count'])
    data.append(fid2data[fid]['H_22_count'])
    data.append(fid2data[fid]['H_23_count'])
    
    for query_file in query_files:
        date = query_file[:4]
        data.append(fid2data[fid]['date_'+date+'_count'])
    
    for pid in pids:
        data.append(fid2data[fid]['pid_'+pid+'_count'])
    
    
#    for query_file in query_files:
#        date = query_file[:4]
#        for pid in pids:
#            data.append(fid2data[fid]['date-pid_'+date+'-'+pid+'_count'])
    
    data.append(fid2data[fid]['qts_step_mean'])
    data.append(fid2data[fid]['qts_step_std'])
    data.append(fid2data[fid]['qts_step_min'])
    data.append(fid2data[fid]['qts_step_Q1'])
    data.append(fid2data[fid]['qts_step_Q2'])
    data.append(fid2data[fid]['qts_step_Q3'])
    data.append(fid2data[fid]['qts_step_max'])
    
    data.append(fid2data[fid]['count'])
    
    data.append(fid2data[fid]['y'])
    dataset.append(data)

f = open('output.csv', 'w')
w = csv.writer(f)
w.writerows(dataset)
f.close()

print('Time Taken:', time.time()-stime)
    