import csv, pickle, os
import pandas as pd

def getPath():

    root_dir = '.'
    data_dir = os.path.join(root_dir, 'data')
    info_dir = os.path.join(root_dir, 'info')
    info_pkl_dir = os.path.join(info_dir, 'pkl')
    info_csv_dir = os.path.join(info_dir, 'csv')
    exception_dir = os.path.join(data_dir, 'exception')
    query_dir = os.path.join(data_dir, 'query_log')
    # query_dir = os.path.join(data_dir, 'small_query_log')

    exception_training_file = os.path.join(exception_dir, 'exception_train.txt')
    exception_test_file = os.path.join(exception_dir, 'exception_testing.txt')

    training_set_file = os.path.join(data_dir, 'training-set.csv') 
    testing_set_file = os.path.join(data_dir, 'testing-set.csv')

    # output info 
    # malware_FID_PKL_file = os.path.join(info_dir, 'malware_fid.pkl')
    malware_FID_CSV_file = os.path.join(info_csv_dir, 'malware_fid.csv')
    normal_FID_CSV_file = os.path.join(info_csv_dir, 'normal_fid.csv')
    
    train_FID_malware_rate = os.path.join(info_csv_dir, 'train_fid_malware_rate')
    test_FID_malware_rate = os.path.join(info_csv_dir, 'test_fid_malware_rate')
    
    time_feature_csv_file = os.path.join(info_csv_dir, 'time_feature.csv')
    
    id_FID_pkl_file = os.path.join(info_pkl_dir, 'id_fid.pkl')
    FID_id_pkl_file = os.path.join(info_pkl_dir, 'fid_id.pkl')
    
    id_time_feature_pkl_file = os.path.join(info_pkl_dir, 'id_time_feature.pkl')
    time_feature_id_pkl_file = os.path.join(info_pkl_dir, 'time_feature_id.pkl')
    
    tmp = os.path.join(info_csv_dir, 'tmp.txt')

    path = {
        'ROOT_DIR': root_dir,
        'DATA_DIR': data_dir,
        'INFO_DIR': info_dir,
        'EXCEPTION_DIR': exception_dir,
        'QUERY_DIR': query_dir,
        'EXCEPTION_TRAINING_FILE': exception_training_file,
        'EXCEPTION_TESTING_FILE': exception_test_file,
        'TRAINING_SET_FILE': training_set_file,
        'TESTING_SET_FILE': testing_set_file,
        # output info csv and pkl
        # 'MALWARE_FID_PKL_FILE': malware_FID_PKL_file,
        'MALWARE_FID_CSV_FILE': malware_FID_CSV_file,
        'NORMAL_FID_CSV_FILE': normal_FID_CSV_file,
        'TRAIN_FID_MALWARE_RATE': train_FID_malware_rate,
        'TEST_FID_MALWARE_RATE': test_FID_malware_rate,
        'TMP': tmp,
        'TIME_FEATURE_CSV_FILE': time_feature_csv_file,
        'ID_FID_PKL_FILE': id_FID_pkl_file,
        'FID_ID_PKL_FILE': FID_id_pkl_file,
        'ID_TIME_FEATURE_PKL_FILE': id_time_feature_pkl_file,
        'TIME_FEATURE_ID_PKL_FILE': time_feature_id_pkl_file,
    }
    return path

# def writeCSVandPKL(df, csvFile, pklFile):


def readCSV(file):
    # return 2d array of string
    data = []
    with open(file, 'r') as f:
        for row in csv.reader(f):
            data.append(row)
    return data

def writeCSV(data, file):
    # data is 2d array
    with open(file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for line in data:
            writer.writerow(line)
    return

def readPickle(fin):
	with open(fin, 'rb') as file:
		content = pickle.load(file)
	return content

def writePickle(data, fout):
	with open(fout, 'wb') as file:
		pickle.dump(data, file)
	return
