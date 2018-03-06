from utils import *
import pandas as pd

path = getPath()

def getMalwareFID():
    df = pd.read_csv(path['TRAINING_SET_FILE'], header=None)
    malwareFID = df.loc[df[1] == 1][[0]]
    normalFID = df.loc[df[1] == 0][[0]]
    malwareFID.to_csv(path['MALWARE_FID_CSV_FILE'], sep=',', header=None, index=None)
    normalFID.to_csv(path['NORMAL_FID_CSV_FILE'], sep=',', header=None, index=None)

    return normalFID, malwareFID

def getSeperateTimeNumFID(timeNumFID, query_df, fid):
    
    query_df['timestamp'] = pd.to_datetime(query_df['timestamp'], unit='s') 
    gb = query_df.groupby('fid')
    print(query_df)
    if type(timeNumFID['totalNum']) != type(None):
        print('add')
        timeNumFID['totalNum'] = timeNumFID['totalNum'].add(gb.size(), fill_value=0)
        
    else:
        print('create')
        timeNumFID['totalNum'] = gb.size()
    # print(gb.get_group('05d8dd88fa342850d7245430ceed39bb'))
    # if type(timeNumFID['monethlyNum']) != type(None):
       

    # month = 

    # timeNumFID['totalNum'] += gb.size()
    # print(gb.size())
    print('==================')
    for key, item in gb:
        print( gb.get_group(key)['timestamp'].dt.month)
        
        print( gb.get_group(key)['timestamp'].dt.month.count())
        print( type(gb.get_group(key)))
        break
        
    # print(query_df)
    # query_df[2].dt.strftime('%Y-%m-%d %H:%M:%S')
    # group by FID
    # group by month
    # group by hour
    # query_df[2].hour.groupby(df[2])
    # print(query_df[2])
    # print(type(query_df[2][0]))
    
    return timeNumFID

def getSeperateDeviceNumFID():
    return []
def getSeperateCustomerNumFID():
    return []
def getNumFID(normalFID, malwareFID):
    
    # initial time data
    normalTimeNumFID = {
        'totalNum': None,
        'monthlyNum': None,
        'weeklyNum': None,
        'dailyNum': None,
        'hourlyNum': None,
    }
    malwareTimeNumFID = {
        'totalNum': None,
        'monthlyNum': None,
        'weeklyNum': None,
        'dailyNum': None,
        'hourlyNum': None,
    }
    # initial device data
    normalDeviceNumFID = malwareDeviceNumFID = {
        'deviceNum': None,
    
    }
    # initial customer data
    normalCustomerFID = malwareCustomerFID = {
        'customerNum': None,

    } 

    for filename in os.listdir(path['QUERY_DIR']):
        if filename == '0406.csv':
            continue
        print('processing file : %s' % (filename))
        query_df = pd.read_csv(os.path.join(path['QUERY_DIR'], filename), header=None, names=['fid', 'cid', 'timestamp', 'did'])
        # print(query_df)
        # print('=======================')
        # print(query_df['fid'])
        # print(malwareFID[0].tolist())
        
        # print(malwareFID[0].tolist())
        # print('-------------------------------------------')
        normal_query_df = query_df[query_df['fid'].isin(normalFID[0].tolist())]
        malware_query_df = query_df[query_df['fid'].isin(malwareFID[0].tolist())]
        # print(malware_query_df)
        # print(normal_query_df)
        if normal_query_df.empty and malware_query_df.empty:
            print('empty : %s' % (filename))
            continue        
        else:
            print('not empty')

        normalTimeNumFID = getSeperateTimeNumFID(normalTimeNumFID, normal_query_df, normalFID)
        malwareTimeNumFID = getSeperateTimeNumFID(malwareTimeNumFID, malware_query_df, malwareFID)

        normalDeviceNumFID = getSeperateDeviceNumFID()
        malwareDeviceNumFID = getSeperateDeviceNumFID()

        normalCustomerFID = getSeperateCustomerNumFID()
        malwareCustomerFID = getSeperateCustomerNumFID()
        print('malware size : ', end=' ')
        print(malwareTimeNumFID['totalNum'].size)
        print('normal size : ', end=' ')
        print(normalTimeNumFID['totalNum'].size)
        break

        # print('======================= mal ==========================')
        # print(malwareTimeNumFID['totalNum'])
        print('=====================================================')

        # if filename == '0507.csv':
        #     break
    print(malwareTimeNumFID['totalNum'])
    return [], []

def main():
    
    print('[ STAGE 1 ] : seperate FID')
    normalFID, malwareFID = getMalwareFID()
    
    print('[ STAGE 2 ] : get total, monthly, weekly, daily and hourly data of FID')
    normalTimeNumFID, malwareTimeNumFID = getNumFID(normalFID, malwareFID)
    # normalNumFID = getNumFID(normalFID, path['NORMAL_FID_CSV_FILE'])
    # malwareNumFID = getNumFID(malwareFID, path['MALWARE_FID_CSV_FILE'])

    print('[ STAGE 3 ] : ')


if __name__ == '__main__':
    
    main()