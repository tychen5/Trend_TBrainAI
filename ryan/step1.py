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

eps = 10 ** -5


"""
fids = set()
cids = set()
pids = set()

for date_file in query_files:
    print(date_file)
    f = open(os.path.join(query_log_dir, date_file), 'r')
    for line in csv.reader(f):
        
        fid = line[0]
        cid = line[1]
#        qts = int(line[2])
        pid = line[3]
        
        fids.add(fid)
        cids.add(cid)
        pids.add(pid)


f = open('fids.txt', 'w')
for fid in sorted(fids):
    f.write(fid + '\n')
f.close()


f = open('cids.txt', 'w')
for cid in sorted(cids):
    f.write(cid + '\n')
f.close()


f = open('pids.txt', 'w')
for pid in sorted(pids):
    f.write(pid + '\n')
f.close()

"""


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

fids = sorted(list(fids))
cids = sorted(list(cids))
pids = sorted(list(pids))



fid2data = {}




for i in range(len(fids)):
    
    if i % 100 == 0:
        print('Dict Set', i, '/', len(fids))
    
    fid = fids[i]
    
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
    
    
    
    fid2data[fid]['H_00_ratio'] = 0
    fid2data[fid]['H_01_ratio'] = 0
    fid2data[fid]['H_02_ratio'] = 0
    fid2data[fid]['H_03_ratio'] = 0
    fid2data[fid]['H_04_ratio'] = 0
    fid2data[fid]['H_05_ratio'] = 0
    fid2data[fid]['H_06_ratio'] = 0
    fid2data[fid]['H_07_ratio'] = 0
    fid2data[fid]['H_08_ratio'] = 0
    fid2data[fid]['H_09_ratio'] = 0
    fid2data[fid]['H_10_ratio'] = 0
    fid2data[fid]['H_11_ratio'] = 0
    fid2data[fid]['H_12_ratio'] = 0
    fid2data[fid]['H_13_ratio'] = 0
    fid2data[fid]['H_14_ratio'] = 0
    fid2data[fid]['H_15_ratio'] = 0
    fid2data[fid]['H_16_ratio'] = 0
    fid2data[fid]['H_17_ratio'] = 0
    fid2data[fid]['H_18_ratio'] = 0
    fid2data[fid]['H_19_ratio'] = 0
    fid2data[fid]['H_20_ratio'] = 0
    fid2data[fid]['H_21_ratio'] = 0
    fid2data[fid]['H_22_ratio'] = 0
    fid2data[fid]['H_23_ratio'] = 0
    
    
    
    
    for query_file in query_files:
        date = query_file[:4]
        fid2data[fid]['date_'+date+'_count'] = 0
    for query_file in query_files:
        date = query_file[:4]
        fid2data[fid]['date_'+date+'_ratio'] = 0
    
    for i in range(1, len(query_files)):
        pre_date = query_files[i-1][:4]
        date = query_files[i][:4]
        fid2data[fid]['date_'+date+'/'+pre_date+'_ratio'] = 0
    
    
    
    for i in range(7):
        fid2data[fid]['appear_day_'+str(i+1)+'_count'] = 0
    for i in range(7):
        fid2data[fid]['appear_day_'+str(i+1)+'_ratio'] = 0
    
    fid2data[fid]['appear_day_2_d/p_ratio'] = 0
    fid2data[fid]['appear_day_3_d/p_ratio'] = 0
    fid2data[fid]['appear_day_4_d/p_ratio'] = 0
    fid2data[fid]['appear_day_5_d/p_ratio'] = 0
    fid2data[fid]['appear_day_6_d/p_ratio'] = 0
    fid2data[fid]['appear_day_7_d/p_ratio'] = 0
        
        
    for pid in pids:
        fid2data[fid]['pid_'+pid+'_count'] = 0
    for pid in pids:
        fid2data[fid]['pid_'+pid+'_ratio'] = 0
    
    
    fid2data[fid]['prior1_cid'] = ''
    fid2data[fid]['prior1_cid_match_count'] = 0
    fid2data[fid]['prior1_cid_match_ratio'] = 0
    
    fid2data[fid]['prior2_cid'] = ''
    fid2data[fid]['prior2_cid_match_count'] = 0
    fid2data[fid]['prior2_cid_match_ratio'] = 0
    
    fid2data[fid]['prior3_cid'] = ''
    fid2data[fid]['prior3_cid_match_count'] = 0
    fid2data[fid]['prior3_cid_match_ratio'] = 0
    
    fid2data[fid]['prior4_cid'] = ''
    fid2data[fid]['prior4_cid_match_count'] = 0
    fid2data[fid]['prior4_cid_match_ratio'] = 0
    
    fid2data[fid]['prior5_cid'] = ''
    fid2data[fid]['prior5_cid_match_count'] = 0
    fid2data[fid]['prior5_cid_match_ratio'] = 0
    
    
    for month in ['march', 'april', 'may']:
        
        fid2data[fid][month + '_qts_list'] = []
        fid2data[fid][month + '_qts_step_list'] = []
        fid2data[fid][month + '_qts_step_mean'] = 0
        fid2data[fid][month + '_qts_step_std'] = 0
        fid2data[fid][month + '_qts_step_min'] = 0
        fid2data[fid][month + '_qts_step_Q0.5'] = 0
        fid2data[fid][month + '_qts_step_Q1'] = 0
        fid2data[fid][month + '_qts_step_Q1.5'] = 0
        fid2data[fid][month + '_qts_step_Q2'] = 0
        fid2data[fid][month + '_qts_step_Q2.5'] = 0
        fid2data[fid][month + '_qts_step_Q3'] = 0
        fid2data[fid][month + '_qts_step_Q3.5'] = 0
        fid2data[fid][month + '_qts_step_max'] = 0
    
    
    fid2data[fid]['qts_list'] = []
    fid2data[fid]['qts_step_list'] = []
    fid2data[fid]['qts_step_mean'] = 0
    fid2data[fid]['qts_step_std'] = 0
    fid2data[fid]['qts_step_min'] = 0
    fid2data[fid]['qts_step_Q0.5'] = 0
    fid2data[fid]['qts_step_Q1'] = 0
    fid2data[fid]['qts_step_Q1.5'] = 0
    fid2data[fid]['qts_step_Q2'] = 0
    fid2data[fid]['qts_step_Q2.5'] = 0
    fid2data[fid]['qts_step_Q3'] = 0
    fid2data[fid]['qts_step_Q3.5'] = 0
    fid2data[fid]['qts_step_max'] = 0
        
    for month in ['march', 'april', 'may']:
        fid2data[fid][month + '_count'] = 0
    for month in ['march', 'april', 'may']:
        fid2data[fid][month + '_ratio'] = 0
    
    fid2data[fid]['count'] = 0
    
    fid2data[fid]['y'] = None
    
    fid2data[fid]['train/test'] = None
    
    






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
        fid2data[line[0]]['train/test'] = 'train'
    else:
#        fid2data[line[0]]['train/test'] = 'train-except'
        continue
        
f.close()
train_fids = sorted(train_fids)


test_fids = set()
f = open(test_path, 'r')
for line in csv.reader(f):
    if line[0] not in test_excepts:
        test_fids.add(line[0])
        fid2data[line[0]]['train/test'] = 'test'
    else:
#        fid2data[line[0]]['train/test'] = 'test-except'
        continue
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
    
    print('Query File Loading:', date_file)
    
    f = open(os.path.join(query_log_dir, date_file), 'r')
    
    for line in csv.reader(f):
        
        fid = line[0]
        cid = line[1]
        qts = int(line[2])
        pid = line[3]
        
        the_time = datetime.datetime.utcfromtimestamp(qts)
        
        the_time_m = the_time.strftime('%m')
        the_time_H = the_time.strftime('%H')
        the_time_date = the_time.strftime('%m%d')
        
        fid2data[fid]['H_'+the_time_H+'_count'] += 1
        fid2data[fid]['date_'+the_time_date+'_count'] += 1
        
        fid2data[fid]['pid_'+pid+'_count'] += 1
        
        fid2data[fid]['qts_list'].append(qts)
        
        fid2data[fid]['count'] += 1
        
        if the_time_m == '03':
            fid2data[fid]['march_qts_list'].append(qts)
            fid2data[fid]['march_count'] += 1
        elif the_time_m == '04':
            fid2data[fid]['april_qts_list'].append(qts)
            fid2data[fid]['april_count'] += 1
        elif the_time_m == '05':
            fid2data[fid]['may_qts_list'].append(qts)
            fid2data[fid]['may_count'] += 1
        
        if cid == fid2data[fid]['prior1_cid']:
            fid2data[fid]['prior1_cid_match_count'] += 1
        
        if cid == fid2data[fid]['prior2_cid']:
            fid2data[fid]['prior2_cid_match_count'] += 1
        
        if cid == fid2data[fid]['prior3_cid']:
            fid2data[fid]['prior3_cid_match_count'] += 1
        
        if cid == fid2data[fid]['prior4_cid']:
            fid2data[fid]['prior4_cid_match_count'] += 1
        
        if cid == fid2data[fid]['prior5_cid']:
            fid2data[fid]['prior5_cid_match_count'] += 1
        
        
        fid2data[fid]['prior5_cid'] = fid2data[fid]['prior4_cid']
        fid2data[fid]['prior4_cid'] = fid2data[fid]['prior3_cid']
        fid2data[fid]['prior3_cid'] = fid2data[fid]['prior2_cid']
        fid2data[fid]['prior2_cid'] = fid2data[fid]['prior1_cid']
        fid2data[fid]['prior1_cid'] = cid
        
            
        
    f.close()




print('Date/Pre-Date Counting...')
for fid in fids:
    for i in range(1, len(query_files)):
        pre_date = query_files[i-1][:4]
        date = query_files[i][:4]
        fid2data[fid]['date_'+date+'/'+pre_date+'_ratio'] = fid2data[fid]['date_'+date+'_count']/(fid2data[fid]['date_'+pre_date+'_count'] + eps)




print('QTS List Building...')
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
    
    
    for month in ['march', 'april', 'may']:
    
        fid2data[fid][month + '_qts_list'] = sorted(fid2data[fid][month + '_qts_list']) 
        for i in range(len(fid2data[fid][month + '_qts_list'])):
            if i == 0:
                continue
            else:
                the_qts_step = fid2data[fid][month + '_qts_list'][i] - fid2data[fid][month + '_qts_list'][i - 1]
                if the_qts_step < 0:
                    sys.exit()
                fid2data[fid][month + '_qts_step_list'].append(the_qts_step)
    


print('QTS Step List Building...')
for fid in fids:
    
    the_qts_step_list = fid2data[fid]['qts_step_list']
    
    if not the_qts_step_list:
        continue
    
    fid2data[fid]['qts_step_mean'] = np.mean(the_qts_step_list)
    fid2data[fid]['qts_step_std'] = np.std(the_qts_step_list)
    fid2data[fid]['qts_step_min'] = np.min(the_qts_step_list)
    fid2data[fid]['qts_step_Q0.5'] = np.percentile(the_qts_step_list, 12.5)
    fid2data[fid]['qts_step_Q1'] = np.percentile(the_qts_step_list, 25)
    fid2data[fid]['qts_step_Q1.5'] = np.percentile(the_qts_step_list, 37.5)
    fid2data[fid]['qts_step_Q2'] = np.percentile(the_qts_step_list, 50)
    fid2data[fid]['qts_step_Q2.5'] = np.percentile(the_qts_step_list, 62.5)
    fid2data[fid]['qts_step_Q3'] = np.percentile(the_qts_step_list, 75)
    fid2data[fid]['qts_step_Q3.5'] = np.percentile(the_qts_step_list, 87.5)
    fid2data[fid]['qts_step_max'] = np.max(the_qts_step_list)
    
    
    for month in ['march', 'april', 'may']:
        
    
        the_qts_step_list = fid2data[fid][month + '_qts_step_list']
        
        if not the_qts_step_list:
            continue
        
        fid2data[fid][month + '_qts_step_mean'] = np.mean(the_qts_step_list)
        fid2data[fid][month + '_qts_step_std'] = np.std(the_qts_step_list)
        fid2data[fid][month + '_qts_step_min'] = np.min(the_qts_step_list)
        fid2data[fid][month + '_qts_step_Q0.5'] = np.percentile(the_qts_step_list, 12.5)
        fid2data[fid][month + '_qts_step_Q1'] = np.percentile(the_qts_step_list, 25)
        fid2data[fid][month + '_qts_step_Q1.5'] = np.percentile(the_qts_step_list, 37.5)
        fid2data[fid][month + '_qts_step_Q2'] = np.percentile(the_qts_step_list, 50)
        fid2data[fid][month + '_qts_step_Q2.5'] = np.percentile(the_qts_step_list, 62.5)
        fid2data[fid][month + '_qts_step_Q3'] = np.percentile(the_qts_step_list, 75)
        fid2data[fid][month + '_qts_step_Q3.5'] = np.percentile(the_qts_step_list, 87.5)
        fid2data[fid][month + '_qts_step_max'] = np.max(the_qts_step_list)



print('Ratio Counting...')
for fid in fids:
    
    fid2data[fid]['H_00_ratio'] = fid2data[fid]['H_00_count']/fid2data[fid]['count']
    fid2data[fid]['H_01_ratio'] = fid2data[fid]['H_01_count']/fid2data[fid]['count']
    fid2data[fid]['H_02_ratio'] = fid2data[fid]['H_02_count']/fid2data[fid]['count']
    fid2data[fid]['H_03_ratio'] = fid2data[fid]['H_03_count']/fid2data[fid]['count']
    fid2data[fid]['H_04_ratio'] = fid2data[fid]['H_04_count']/fid2data[fid]['count']
    fid2data[fid]['H_05_ratio'] = fid2data[fid]['H_05_count']/fid2data[fid]['count']
    fid2data[fid]['H_06_ratio'] = fid2data[fid]['H_06_count']/fid2data[fid]['count']
    fid2data[fid]['H_07_ratio'] = fid2data[fid]['H_07_count']/fid2data[fid]['count']
    fid2data[fid]['H_08_ratio'] = fid2data[fid]['H_08_count']/fid2data[fid]['count']
    fid2data[fid]['H_09_ratio'] = fid2data[fid]['H_09_count']/fid2data[fid]['count']
    fid2data[fid]['H_10_ratio'] = fid2data[fid]['H_10_count']/fid2data[fid]['count']
    fid2data[fid]['H_11_ratio'] = fid2data[fid]['H_11_count']/fid2data[fid]['count']
    fid2data[fid]['H_12_ratio'] = fid2data[fid]['H_12_count']/fid2data[fid]['count']
    fid2data[fid]['H_13_ratio'] = fid2data[fid]['H_13_count']/fid2data[fid]['count']
    fid2data[fid]['H_14_ratio'] = fid2data[fid]['H_14_count']/fid2data[fid]['count']
    fid2data[fid]['H_15_ratio'] = fid2data[fid]['H_15_count']/fid2data[fid]['count']
    fid2data[fid]['H_16_ratio'] = fid2data[fid]['H_16_count']/fid2data[fid]['count']
    fid2data[fid]['H_17_ratio'] = fid2data[fid]['H_17_count']/fid2data[fid]['count']
    fid2data[fid]['H_18_ratio'] = fid2data[fid]['H_18_count']/fid2data[fid]['count']
    fid2data[fid]['H_19_ratio'] = fid2data[fid]['H_19_count']/fid2data[fid]['count']
    fid2data[fid]['H_20_ratio'] = fid2data[fid]['H_20_count']/fid2data[fid]['count']
    fid2data[fid]['H_21_ratio'] = fid2data[fid]['H_21_count']/fid2data[fid]['count']
    fid2data[fid]['H_22_ratio'] = fid2data[fid]['H_22_count']/fid2data[fid]['count']
    fid2data[fid]['H_23_ratio'] = fid2data[fid]['H_23_count']/fid2data[fid]['count']
    
    
    for query_file in query_files:
        date = query_file[:4]
        fid2data[fid]['date_'+date+'_ratio'] = fid2data[fid]['date_'+date+'_count']/fid2data[fid]['count']
    
    
    for pid in pids:
        fid2data[fid]['pid_'+pid+'_ratio'] = fid2data[fid]['pid_'+pid+'_count']/fid2data[fid]['count']
    
    fid2data[fid]['prior1_cid_match_ratio'] = fid2data[fid]['prior1_cid_match_count']/fid2data[fid]['count']
    fid2data[fid]['prior2_cid_match_ratio'] = fid2data[fid]['prior2_cid_match_count']/fid2data[fid]['count']
    fid2data[fid]['prior3_cid_match_ratio'] = fid2data[fid]['prior3_cid_match_count']/fid2data[fid]['count']
    fid2data[fid]['prior4_cid_match_ratio'] = fid2data[fid]['prior4_cid_match_count']/fid2data[fid]['count']
    fid2data[fid]['prior5_cid_match_ratio'] = fid2data[fid]['prior5_cid_match_count']/fid2data[fid]['count']
    
    for month in ['march', 'april', 'may']:
        fid2data[fid][month + '_ratio'] = fid2data[fid][month + '_count']/fid2data[fid]['count']






print('Pop Windows Building...')
for fid in fids:
    
    for i in range(len(query_files)):
        
        date = query_files[i][:4]
        
        if fid2data[fid]['date_'+date+'_count'] > 0:
            
            for k in range(7):
                if i+k < len(query_files): 
                    fid2data[fid]['appear_day_'+str(k+1)+'_count'] = fid2data[fid]['date_'+query_files[i+k][:4]+'_count']
                    fid2data[fid]['appear_day_'+str(k+1)+'_ratio'] = fid2data[fid]['date_'+query_files[i+k][:4]+'_ratio']
                else:
                    break
            
            fid2data[fid]['appear_day_2_d/p_ratio'] = fid2data[fid]['appear_day_2_count']/(fid2data[fid]['appear_day_1_count'] + eps)
            fid2data[fid]['appear_day_3_d/p_ratio'] = fid2data[fid]['appear_day_3_count']/(fid2data[fid]['appear_day_2_count'] + eps)
            fid2data[fid]['appear_day_4_d/p_ratio'] = fid2data[fid]['appear_day_4_count']/(fid2data[fid]['appear_day_3_count'] + eps)
            fid2data[fid]['appear_day_5_d/p_ratio'] = fid2data[fid]['appear_day_5_count']/(fid2data[fid]['appear_day_4_count'] + eps)
            fid2data[fid]['appear_day_6_d/p_ratio'] = fid2data[fid]['appear_day_6_count']/(fid2data[fid]['appear_day_5_count'] + eps)
            fid2data[fid]['appear_day_7_d/p_ratio'] = fid2data[fid]['appear_day_7_count']/(fid2data[fid]['appear_day_6_count'] + eps)
            
            break
            
        
    
print('Dict to List of List...')

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


data.append('H_00_ratio')
data.append('H_01_ratio')
data.append('H_02_ratio')
data.append('H_03_ratio')
data.append('H_04_ratio')
data.append('H_05_ratio')
data.append('H_06_ratio')
data.append('H_07_ratio')
data.append('H_08_ratio')
data.append('H_09_ratio')
data.append('H_10_ratio')
data.append('H_11_ratio')
data.append('H_12_ratio')
data.append('H_13_ratio')
data.append('H_14_ratio')
data.append('H_15_ratio')
data.append('H_16_ratio')
data.append('H_17_ratio')
data.append('H_18_ratio')
data.append('H_19_ratio')
data.append('H_20_ratio')
data.append('H_21_ratio')
data.append('H_22_ratio')
data.append('H_23_ratio')


for query_file in query_files:
    date = query_file[:4]
    data.append('date_'+date+'_count')
for query_file in query_files:
    date = query_file[:4]
    data.append('date_'+date+'_ratio')


for i in range(1, len(query_files)):
    pre_date = query_files[i-1][:4]
    date = query_files[i][:4]
    data.append('date_'+date+'/'+pre_date+'_ratio')



for i in range(7):
    data.append('appear_day_'+str(i+1)+'_count')
for i in range(7):
    data.append('appear_day_'+str(i+1)+'_ratio')



data.append('appear_day_2_d/p_ratio')
data.append('appear_day_3_d/p_ratio')
data.append('appear_day_4_d/p_ratio')
data.append('appear_day_5_d/p_ratio')
data.append('appear_day_6_d/p_ratio')
data.append('appear_day_7_d/p_ratio')


for pid in pids:
    data.append('pid_'+pid+'_count')
for pid in pids:
    data.append('pid_'+pid+'_ratio')

data.append('prior1_cid_match_count')
data.append('prior1_cid_match_ratio')

data.append('prior2_cid_match_count')
data.append('prior2_cid_match_ratio')

data.append('prior3_cid_match_count')
data.append('prior3_cid_match_ratio')

data.append('prior4_cid_match_count')
data.append('prior4_cid_match_ratio')

data.append('prior5_cid_match_count')
data.append('prior5_cid_match_ratio')

for month in ['march', 'april', 'may']:
    data.append(month + '_qts_step_mean')
    data.append(month + '_qts_step_std')
    data.append(month + '_qts_step_min')
    data.append(month + '_qts_step_Q0.5')
    data.append(month + '_qts_step_Q1')
    data.append(month + '_qts_step_Q1.5')
    data.append(month + '_qts_step_Q2')
    data.append(month + '_qts_step_Q2.5')
    data.append(month + '_qts_step_Q3')
    data.append(month + '_qts_step_Q3.5')
    data.append(month + '_qts_step_max')


data.append('qts_step_mean')
data.append('qts_step_std')
data.append('qts_step_min')
data.append('qts_step_Q0.5')
data.append('qts_step_Q1')
data.append('qts_step_Q1.5')
data.append('qts_step_Q2')
data.append('qts_step_Q2.5')
data.append('qts_step_Q3')
data.append('qts_step_Q3.5')
data.append('qts_step_max')

    


for month in ['march', 'april', 'may']:
    data.append(month + '_count')
for month in ['march', 'april', 'may']:
    data.append(month + '_ratio')
    

data.append('count')

data.append('y')

data.append('train/test')

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
    
    
    data.append(fid2data[fid]['H_00_ratio'])
    data.append(fid2data[fid]['H_01_ratio'])
    data.append(fid2data[fid]['H_02_ratio'])
    data.append(fid2data[fid]['H_03_ratio'])
    data.append(fid2data[fid]['H_04_ratio'])
    data.append(fid2data[fid]['H_05_ratio'])
    data.append(fid2data[fid]['H_06_ratio'])
    data.append(fid2data[fid]['H_07_ratio'])
    data.append(fid2data[fid]['H_08_ratio'])
    data.append(fid2data[fid]['H_09_ratio'])
    data.append(fid2data[fid]['H_10_ratio'])
    data.append(fid2data[fid]['H_11_ratio'])
    data.append(fid2data[fid]['H_12_ratio'])
    data.append(fid2data[fid]['H_13_ratio'])
    data.append(fid2data[fid]['H_14_ratio'])
    data.append(fid2data[fid]['H_15_ratio'])
    data.append(fid2data[fid]['H_16_ratio'])
    data.append(fid2data[fid]['H_17_ratio'])
    data.append(fid2data[fid]['H_18_ratio'])
    data.append(fid2data[fid]['H_19_ratio'])
    data.append(fid2data[fid]['H_20_ratio'])
    data.append(fid2data[fid]['H_21_ratio'])
    data.append(fid2data[fid]['H_22_ratio'])
    data.append(fid2data[fid]['H_23_ratio'])
    
    for query_file in query_files:
        date = query_file[:4]
        data.append(fid2data[fid]['date_'+date+'_count'])
    for query_file in query_files:
        date = query_file[:4]
        data.append(fid2data[fid]['date_'+date+'_ratio'])
        
    
    for i in range(1, len(query_files)):
        pre_date = query_files[i-1][:4]
        date = query_files[i][:4]
        data.append(fid2data[fid]['date_'+date+'/'+pre_date+'_ratio'])
    
    
    
    for i in range(7):
        data.append(fid2data[fid]['appear_day_'+str(i+1)+'_count'])
    for i in range(7):
        data.append(fid2data[fid]['appear_day_'+str(i+1)+'_ratio'])
    
    
    
    data.append(fid2data[fid]['appear_day_2_d/p_ratio'])
    data.append(fid2data[fid]['appear_day_3_d/p_ratio'])
    data.append(fid2data[fid]['appear_day_4_d/p_ratio'])
    data.append(fid2data[fid]['appear_day_5_d/p_ratio'])
    data.append(fid2data[fid]['appear_day_6_d/p_ratio'])
    data.append(fid2data[fid]['appear_day_7_d/p_ratio'])
    
    
    
    for pid in pids:
        data.append(fid2data[fid]['pid_'+pid+'_count'])
    for pid in pids:
        data.append(fid2data[fid]['pid_'+pid+'_ratio'])
        
    
    data.append(fid2data[fid]['prior1_cid_match_count'])
    data.append(fid2data[fid]['prior1_cid_match_ratio'])
    
    data.append(fid2data[fid]['prior2_cid_match_count'])
    data.append(fid2data[fid]['prior2_cid_match_ratio'])
    
    data.append(fid2data[fid]['prior3_cid_match_count'])
    data.append(fid2data[fid]['prior3_cid_match_ratio'])
    
    data.append(fid2data[fid]['prior4_cid_match_count'])
    data.append(fid2data[fid]['prior4_cid_match_ratio'])
    
    data.append(fid2data[fid]['prior5_cid_match_count'])
    data.append(fid2data[fid]['prior5_cid_match_ratio'])
        
    for month in ['march', 'april', 'may']:
        data.append(fid2data[fid][month + '_qts_step_mean'])
        data.append(fid2data[fid][month + '_qts_step_std'])
        data.append(fid2data[fid][month + '_qts_step_min'])
        data.append(fid2data[fid][month + '_qts_step_Q0.5'])
        data.append(fid2data[fid][month + '_qts_step_Q1'])
        data.append(fid2data[fid][month + '_qts_step_Q1.5'])
        data.append(fid2data[fid][month + '_qts_step_Q2'])
        data.append(fid2data[fid][month + '_qts_step_Q2.5'])
        data.append(fid2data[fid][month + '_qts_step_Q3'])
        data.append(fid2data[fid][month + '_qts_step_Q3.5'])
        data.append(fid2data[fid][month + '_qts_step_max'])
    
    data.append(fid2data[fid]['qts_step_mean'])
    data.append(fid2data[fid]['qts_step_std'])
    data.append(fid2data[fid]['qts_step_min'])
    data.append(fid2data[fid]['qts_step_Q0.5'])
    data.append(fid2data[fid]['qts_step_Q1'])
    data.append(fid2data[fid]['qts_step_Q1.5'])
    data.append(fid2data[fid]['qts_step_Q2'])
    data.append(fid2data[fid]['qts_step_Q2.5'])
    data.append(fid2data[fid]['qts_step_Q3'])
    data.append(fid2data[fid]['qts_step_Q3.5'])
    data.append(fid2data[fid]['qts_step_max'])
    

    for month in ['march', 'april', 'may']:
        data.append(fid2data[fid][month + '_count'])
    for month in ['march', 'april', 'may']:
        data.append(fid2data[fid][month + '_ratio'])
    
    data.append(fid2data[fid]['count'])
    
    data.append(fid2data[fid]['y'])
    
    data.append(fid2data[fid]['train/test'])
    
    dataset.append(data)


print('Output CSV...')

f = open('output.csv', 'w')
w = csv.writer(f)
w.writerows(dataset)
f.close()

print('Time Taken:', time.time()-stime)
    