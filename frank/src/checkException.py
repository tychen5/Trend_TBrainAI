import csv, os, argparse
import numpy as np

from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', help='root directory', default=os.getcwd())
    args = parser.parse_args()
    return args

def getPath(args):

    root_dir = args.directory
    data_dir = os.path.join(root_dir, 'data')
    exception_dir = os.path.join(data_dir, 'exception')
    query_dir = os.path.join(data_dir, 'query_log')

    exception_training_file = os.path.join(exception_dir, 'exception_train.txt')
    exception_test_file = os.path.join(exception_dir, 'exception_testing.txt')

    training_set_file = os.path.join(data_dir, 'training-set.csv') 
    testing_set_file = os.path.join(data_dir, 'testing-set.csv')

    path = {
        'ROOT_DIR': root_dir,
        'DATA_DIR': data_dir,
        'EXCEPTION_DIR': exception_dir,
        'QUERY_DIR': query_dir,
        'EXCEPTION_TRAINING_FILE': exception_training_file,
        'EXCEPTION_TESTING_FILE': exception_test_file,
        'TRAINING_SET_FILE': training_set_file,
        'TESTING_SET_FILE': testing_set_file
    }
    return path

def getException(path):
    excTrainingFileID = np.array(readCSV(path['EXCEPTION_TRAINING_FILE']))[:, 0]
    excTestingFileID = np.array(readCSV(path['EXCEPTION_TESTING_FILE']))[:, 0]
    
    return excTrainingFileID, excTestingFileID

def checkQueryLog(path, excTrainingFileID, excTestingFileID):
    
    excTotalID = np.append(excTrainingFileID, excTestingFileID)
    # print(excTotalID.shape)

    ans = False
    

    for idx, filename in enumerate(os.listdir(path['QUERY_DIR'])):
        
        tmp_ans = False
        file = os.path.join(path['QUERY_DIR'], filename)
        fileID = readCSV(file)
        fileID = np.array(fileID)[:, 0]

        for excID in excTotalID:
            if excID in fileID:
                tmp_ans = True
                print('Query %3d name %10d has exception %s' % (idx, filename, excID))

        


        if tmp_ans == False:
            print('Query %3d name %10s is OK' % (idx, filename))
    
    
    print('checking %d query log' % (len(os.listdir(path['QUERY_DIR']))))

    if ans == True:
        # train set has exception
        print('Exception should be remove.')        
    else:    
        # train set doesn't have exception
        print('OK!')

    return

def checkTrainingSet(path, excTrainingFileID):
    
    ans = False # no exception

    trainingSetID = readCSV(path['TRAINING_SET_FILE'])
    # print(trainingSetID.shape)
    trainingSetID = np.array(trainingSetID)[:, 0]
    # print(trainingSetID.shape)
    for excID in excTrainingFileID:
        if excID in trainingSetID:
            ans = True
            break
        
    # print(trainingSetID[:40])

    print('Checking training-set.csv...')

    if ans == True:
        # train set has exception
        print('Exception should be remove.')        
    else:    
        # train set doesn't have exception
        print('OK!')

    return

def checkTestingSet(path, excTestingFileID):
    
    ans = False # no exception

    testingSetID = readCSV(path['TESTING_SET_FILE'])
    # print(testingSetID.shape)
    testingSetID = np.array(testingSetID)[:, 0]
    # print(testingSetID.shape)
    for excID in excTestingFileID:
        if excID in testingSetID:
            ans = True
            break
        
    # print(testingSetID[:40])

    print('Checking training-set.csv...')

    if ans == True:
        # train set has exception
        print('Exception should be remove.')        
    else:    
        # train set doesn't have exception
        print('OK!')

    return



if __name__ == '__main__':
    args = parse_args()
    path = getPath(args)
    
    excTrainingFileID, excTestingFileID = getException(path)

    checkQueryLog(path, excTrainingFileID, excTestingFileID)
    checkTrainingSet(path, excTrainingFileID)
    checkTestingSet(path, excTestingFileID)
    

    