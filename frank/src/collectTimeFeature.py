from utils import *

import numpy as np
import datetime

path = getPath()

def getColMapping(flag):
    
    if flag == 0:
        # load from mapping file
        print('loading from mapping')
        feature2id = readPickle(path['TIME_FEATURE_ID_PKL_FILE'])
        id2feature = readPickle(path['ID_TIME_FEATURE_PKL_FILE'])
        
    else:
        # load from dataset
        print('generating feature mapping')
        feature2id = {}
        id2feature = {}

    print('total fid size = %d' % (len(feature2id)))

    return feature2id, id2feature, len(feature2id)

def getRowMapping(flag):
    
    if flag == 0:
        # load from mapping file
        print('loading from mapping')
        id2fid = readPickle(path['ID_FID_PKL_FILE'])
        fid2id = readPickle(path['FID_ID_PKL_FILE'])
    else:
        # load from dataset
        print('generating fid mapping')
        
        fid2id = {}
        id2fid = {}
        
        train_fid = np.asarray(readCSV(path['TRAINING_SET_FILE']))[:, 0]
        test_fid = np.asarray(readCSV(path['TESTING_SET_FILE']))[:, 0]
        total_fid = np.concatenate((train_fid, test_fid), axis = 0)
        for id, fid in enumerate(total_fid):
            fid2id[fid] = id
            id2fid[id] = fid
    print('total fid size = %d' % (len(fid2id)))

    return fid2id, id2fid, len(fid2id)

def addFeatureToMatrix(mat, feature2id, id2feature, feature_name, feature_array):
    if feature_name not in feature2id:
        feature2id[feature_name] = len(feature2id)
        id2feature[len(id2feature)] = feature_name
    
    if feature2id[feature_name] >= mat.shape[1]:
        print(feature_array.shape)
        mat = np.concatenate((
            mat, 
            feature_array.reshape(feature_array.shape[0], 1)
            ), 
            axis = 1
        )
    return mat, feature2id, id2feature

def getHourlyFeature(hourly_cnt, fid2id, id2fid, query_fid, query_hour):
    for fid, hour in zip(query_fid, query_hour):
        if fid not in fid2id:
            continue
        else:
            hourly_cnt['H%s_cnt' % (hour)][fid2id[fid]] += 1
    
    return hourly_cnt

def getDailyFeature(daily_cnt, fid2id, id2fid, query_fid, query_day):
    for fid, day in zip(query_fid, query_day):
        if fid not in fid2id:
            continue
        else:
            daily_cnt['D%s_cnt' % (day)][fid2id[fid]] += 1
    
    return daily_cnt

def getMonthlyFeature(monthly_cnt, fid2id, id2fid, query_fid, query_month):
    for fid, month in zip(query_fid, query_month):
        if fid not in fid2id:
            continue
        else:
            monthly_cnt['M%s_cnt' % (month)][fid2id[fid]] += 1
    
    return monthly_cnt

def getTimeFeature(refresh_flag = False):
    
    if refresh_flag or not os.path.exists(path['TIME_FEATURE_CSV_FILE']):
        # collect every time feature
        print('Start collecting time feature')
        
        # get fid id mapping
        flag = 0 if os.path.exists(path['ID_FID_PKL_FILE']) else 1        
        fid2id, id2fid, fid_size = getRowMapping(flag)
        
        # get col feature id mapping
        flag = 0 if os.path.exists(path['ID_TIME_FEATURE_PKL_FILE']) else 1
        feature2id, id2feature, feature_size = getColMapping(flag)

        # get time feature 2d numpy array
        flag = 0 if os.path.exists(path['TIME_FEATURE_CSV_FILE']) else 1
        if os.path.exists(path['TIME_FEATURE_CSV_FILE']):
            time_feature_matrix = np.asarray(readCSV(path['TIME_FEATURE_CSV_FILE']))
        else:
            time_feature_matrix = np.zeros((fid_size, feature_size))
        
        # get time feature
        hourly_cnt = {'H%d_cnt' % (i): np.zeros(fid_size) for i in range(24)}
        daily_cnt = {'D%d_cnt' % (i): np.zeros(fid_size) for i in range(31)}
        monthly_cnt = {'M%d_cnt' % (i): np.zeros(fid_size) for i in [3, 4, 5]}
        
        for query_idx, query_file in enumerate(os.listdir(path['QUERY_DIR'])):
            print('[ %2d ] processing file: %10s' % (query_idx + 1, query_file))
            query = np.asarray(readCSV(os.path.join(path['QUERY_DIR'], query_file)))
            
            query_fid = query[:, 0]
            query_cid = query[:, 1]
            query_timestamp = query[:, 2]
            query_pid = query[:, 3]

            query_dt = np.asarray([datetime.datetime.fromtimestamp(int(timestamp)) for timestamp in query_timestamp ])
            
            # get hourly feature
            query_hour = np.asarray([dt.hour for dt in query_dt])
            hourly_cnt = getHourlyFeature(hourly_cnt, fid2id, id2fid, query_fid, query_hour)
            
            # get daily feature
            query_day = np.asarray([dt.day for dt in query_dt])
            daily_cnt = getDailyFeature(daily_cnt, fid2id, id2fid, query_fid, query_day)
            
            # get monthly feature
            query_month = np.asarray([dt.month for dt in query_dt])
            monthly_cnt = getMonthlyFeature(monthly_cnt, fid2id, id2fid, query_fid, query_month)
            
        # TODO: get other time feature

        # add hourly feature to mat
        for feature_name, feature_array in hourly_cnt.items():
            time_feature_matrix, feature2id, id2feature = addFeatureToMatrix(
                time_feature_matrix,
                feature2id,
                id2feature,
                feature_name, 
                feature_array
            )
        # add daily feature to mat
        for feature_name, feature_array in daily_cnt.items():
            time_feature_matrix, feature2id, id2feature = addFeatureToMatrix(
                time_feature_matrix,
                feature2id,
                id2feature,
                feature_name, 
                feature_array
            )

        # add month feature to mat
        for feature_name, feature_array in monthly_cnt.items():
            time_feature_matrix, feature2id, id2feature = addFeatureToMatrix(
                time_feature_matrix,
                feature2id,
                id2feature,
                feature_name, 
                feature_array
            )

        # TODO: add other feature here


        # store all the information
        writeCSV(time_feature_matrix, path['TIME_FEATURE_CSV_FILE'])
        writePickle(id2fid, path['ID_FID_PKL_FILE'])
        writePickle(fid2id, path['FID_ID_PKL_FILE'])
        writePickle(id2feature, path['ID_TIME_FEATURE_PKL_FILE'])
        writePickle(feature2id, path['TIME_FEATURE_ID_PKL_FILE'])
        
    else:    
        # if file exist and not try to re-collect
        # load file
        time_feature_matrix = np.asarray(readCSV(path['TIME_FEATURE_CSV_FILE']))
        feature2id = readPickle(path['TIME_FEATURE_ID_PKL_FILE'])
        id2feature = readPickle(path['ID_TIME_FEATURE_PKL_FILE'])
    
    return time_feature_matrix, feature2id, id2feature
        
if __name__ == '__main__':
    time_feature_matrix, feature2id, id2feature = getTimeFeature(True)