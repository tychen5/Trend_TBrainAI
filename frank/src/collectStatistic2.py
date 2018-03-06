from utils import *

import os, sys


path = getPath()

def collectAns():
    training_set = readCSV(path['TRAINING_SET_FILE'])
    
    # print(training_set[0])
    FID2ANS = {}
    trainFID_list = []
    for line in training_set:
        FID2ANS[line[0]] = int(line[1])
        trainFID_list.append(line[0])
    
    return FID2ANS, trainFID_list

def getTestList():
    test_set = []
    tmp_test = readCSV(path['TESTING_SET_FILE'])
    for line in tmp_test:
        test_set.append(line[0])
    return test_set


def getMalwareCountCID(FID2ANS):
    malware_count = {}
    FID2CID_dict = {}
    FIDandCIDpair = {}

    for queryFile in os.listdir(path['QUERY_DIR']):
        print('Processing file : %s' % (queryFile))
        query = readCSV(os.path.join(path['QUERY_DIR'], queryFile))
        for query_line in query:
            fid, cid, timestamp, pid = query_line
            
            # get FID2CID_dict
            if fid not in FID2CID_dict:
                FID2CID_dict[fid] = []
            if cid not in FID2CID_dict[fid]:
                FID2CID_dict[fid].append(cid)
            
            # get malware_count 
            if cid not in malware_count:
                malware_count[cid] = {'total_count': 0, 'malware_count': 0}
            if fid in FID2ANS:
                malware_count[cid]['total_count'] += 1
                if FID2ANS[fid] == 1:
                    malware_count[cid]['malware_count'] += 1


            # get FIDandCIDpair
            if cid not in FIDandCIDpair:
                FIDandCIDpair[cid] = {}
            if fid not in FIDandCIDpair[cid]:
                FIDandCIDpair[cid][fid] = 0
            
            FIDandCIDpair[cid][fid] += 1
            

            # print(query_line)
    return malware_count, FID2CID_dict, FIDandCIDpair

def getMalwareCountPID(FID2ANS):
    malware_count = {}
    FID2CID_dict = {}
    FIDandCIDpair = {}

    for queryFile in os.listdir(path['QUERY_DIR']):
        print('Processing file : %s' % (queryFile))
        query = readCSV(os.path.join(path['QUERY_DIR'], queryFile))
        for query_line in query:
            fid, pid, timestamp, cid = query_line
            
            # get FID2CID_dict
            if fid not in FID2CID_dict:
                FID2CID_dict[fid] = []
            if cid not in FID2CID_dict[fid]:
                FID2CID_dict[fid].append(cid)
            
            # get malware_count 
            if cid not in malware_count:
                malware_count[cid] = {'total_count': 0, 'malware_count': 0}
            if fid in FID2ANS:
                malware_count[cid]['total_count'] += 1
                if FID2ANS[fid] == 1:
                    malware_count[cid]['malware_count'] += 1


            # get FIDandCIDpair
            if cid not in FIDandCIDpair:
                FIDandCIDpair[cid] = {}
            if fid not in FIDandCIDpair[cid]:
                FIDandCIDpair[cid][fid] = 0
            
            FIDandCIDpair[cid][fid] += 1
            

            # print(query_line)
    return malware_count, FID2CID_dict, FIDandCIDpair


def getMalwareRateCID(malware_count):
    malware_rate = {}

    aveCIDmalwareCount = {'total_count': 0, 'malware_count': 0}

    for cid, count_dict in malware_count.items():
        if count_dict['malware_count'] != 0:    
            malware_rate[cid] = count_dict['malware_count'] / count_dict['total_count']
            aveCIDmalwareCount['total_count'] += count_dict['total_count']
            aveCIDmalwareCount['malware_count'] += count_dict['malware_count']
    aveCIDmalwareRate = aveCIDmalwareCount['malware_count'] / aveCIDmalwareCount['total_count']

    for cid, count_dict in malware_count.items():
        if count_dict['malware_count'] == 0:
            malware_rate[cid] = aveCIDmalwareRate
    
    return malware_rate

def getMalwareRateFID(FID2CID_dict, FIDandCIDpair, malware_rate_CID):
    # malware_rate_FID = {'weight_sum': 0, 'weight_num': 0}
    malware_rate_FID = {}

    for fid, cid_list in FID2CID_dict.items():
        if fid not in malware_rate_FID:
            malware_rate_FID[fid] = {'weight_sum': 0, 'weight_num': 0, 'weight_ave': 0}
        for cid in cid_list:
            malware_rate_FID[fid]['weight_sum'] += FIDandCIDpair[cid][fid] * malware_rate_CID[cid]
            malware_rate_FID[fid]['weight_num'] += FIDandCIDpair[cid][fid]
        malware_rate_FID[fid]['weight_ave'] = malware_rate_FID[fid]['weight_sum'] / malware_rate_FID[fid]['weight_num']    

    return malware_rate_FID

def getSeperateMalwareRate(fid_list, malware_rate_FID):
    seperate_malware_rate_FID = []
    for fid in fid_list:
        seperate_malware_rate_FID.append([fid, malware_rate_FID[fid]['weight_ave']])
    return seperate_malware_rate_FID

def main(task_flag):
    FID2ANS, trainFID_list = collectAns()
    testFID_list = getTestList()
    # print(FID2ANS.items())
    if task_flag == 0:
        malware_count_CID, FID2CID_dict, FIDandCIDpair = getMalwareCountCID(FID2ANS)
    else:
        malware_count_CID, FID2CID_dict, FIDandCIDpair = getMalwareCountPID(FID2ANS)

    malware_rate_CID = getMalwareRateCID(malware_count_CID)
    cnt = 0
    if task_flag == 0:
        f = open(path['TMP'] + '.cid', 'w')
    else:
        f = open(path['TMP'] + '.pid', 'w')
    f.truncate()    
    
    for cid in malware_rate_CID:
        f.write('cid [ %s ] : ( total, malware, rate ) = ( %5s, %5s, %5s )\n' % (
            cid, 
            malware_count_CID[cid]['total_count'], 
            malware_count_CID[cid]['malware_count'], 
            malware_rate_CID[cid])
        )
        if malware_rate_CID[cid] == 1:
            cnt += 1
    f.write('------------------------------------------------------------------------------------------------\n\n')
    f.write('fid size : %10d, 100%% malware rate size: %10d, 100%% malware rate: %5f\n\n\n' % (len(malware_rate_CID), cnt, cnt / len(malware_rate_CID)))
    malware_rate_FID = getMalwareRateFID(FID2CID_dict, FIDandCIDpair, malware_rate_CID)
    f.write('------------------------------------------------------------------------------------------------\n')
    f.close()
    print(malware_rate_FID)
    
    train_malware_rate_FID = getSeperateMalwareRate(trainFID_list, malware_rate_FID)
    test_malware_rate_FID = getSeperateMalwareRate(testFID_list, malware_rate_FID)

    if task_flag == 0:
        writeCSV(path['TRAIN_FID_MALWARE_RATE'] + '.cid', train_malware_rate_FID)
        writeCSV(path['TEST_FID_MALWARE_RATE'] + '.cid', test_malware_rate_FID)
    else:
        writeCSV(path['TRAIN_FID_MALWARE_RATE'] + '.pid', train_malware_rate_FID)
        writeCSV(path['TEST_FID_MALWARE_RATE'] + '.pid', test_malware_rate_FID)
    return

if __name__ == '__main__':
    print('collect mode %s' % (sys.argv[1]))	    
    main(int(sys.argv[1])) 
    # 0 for cid
    # 1 for pid
    
