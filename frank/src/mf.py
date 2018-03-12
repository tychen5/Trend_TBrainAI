import datetime
import nimfa
import scipy
import numpy as np

from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix

from utils import *
from collectTimeFeature import getRowMapping

# WORKING_CPU = 5
FID_CID_RANK = 10
FID_PID_RANK = 3
MAX_ITER = 10

path = getPath()

def getPairCnt(mapping, xid2id, id2fid):
    row = []
    col = []
    pair_cnt = []
    for id, fid in id2fid.items():
        for xid, cnt in mapping[fid].items():
            row.append(id)
            col.append(xid2id[xid])
            pair_cnt.append(cnt)
    return row, col, pair_cnt

def addToCntMapping(mapping, xid2id, id2xid, query_fid, query_xid):
    # xid means cid or pid in different condition
    for fid, xid in zip(query_fid, query_xid):
        if xid not in xid2id:
            # add to xid id mapping
            xid2id[xid] = len(xid2id)
            id2xid[len(id2xid)] = xid
        if xid not in mapping[fid]:
            mapping[fid][xid] = 0
        else:
            mapping[fid][xid] += 1

    return mapping, xid2id, id2xid
            
    

def getCidAndPidFeatureFromMF(refresh_flag = False):
    if refresh_flag or not os.path.exists(path['MF_FEATURE_CSV_FILE']):
        
        # judge if the basis and coef file exists
        if os.path.exists(path['MF_FID_CID_SPR_MAT_PKL_FILE']) and os.path.exists(path['MF_FID_PID_SPR_MAT_PKL_FILE']):
            print('Loading sparse matrix')
            fid_cid_spr_mat = readPickle(path['MF_FID_CID_SPR_MAT_PKL_FILE'])
            fid_pid_spr_mat = readPickle(path['MF_FID_PID_SPR_MAT_PKL_FILE'])
        else:        
            print('Building sparse matrix')
            flag = 0 if os.path.exists(path['ID_FID_PKL_FILE']) else 1        
            fid2id, id2fid, fid_size = getRowMapping(flag)
            # get test fid
            test_fid = np.asarray(readCSV(path['TESTING_SET_FILE']))[:, 0]
            for fid in test_fid:
                if fid not in fid2id:
                    fid2id[fid] = len(fid2id)
                    id2fid[len(id2fid)] = fid
            fid_size = len(fid2id)

            # cid
            fid_cid_cnt_mapping = {fid: {} for fid in fid2id}
            cid2id = {}
            id2cid = {}
            # pid
            fid_pid_cnt_mapping = {fid: {} for fid in fid2id}
            pid2id = {}
            id2pid = {}

            for query_idx, query_file in enumerate(sorted(os.listdir(path['QUERY_DIR']))):
                print('[ %2d ] processing file: %10s' % (query_idx + 1, query_file))
                
                query = np.asarray(readCSV(os.path.join(path['QUERY_DIR'], query_file)))
                
                query_fid = query[:, 0]
                query_cid = query[:, 1]
                query_timestamp = query[:, 2]
                query_pid = query[:, 3]

                fid_cid_cnt_mapping, cid2id, id2cid = addToCntMapping(fid_cid_cnt_mapping, cid2id, id2cid, query_fid, query_cid)
                fid_pid_cnt_mapping, pid2id, id2pid = addToCntMapping(fid_pid_cnt_mapping, pid2id, id2pid, query_fid, query_pid)    
            
            # get csr_matrix source array
            print('Getting array for building csr_matrix')
            row_fid_cid, col_cid, fid_cid_pair_cnt = getPairCnt(fid_cid_cnt_mapping, cid2id, id2fid)
            row_fid_pid, col_pid, fid_pid_pair_cnt = getPairCnt(fid_pid_cnt_mapping, pid2id, id2fid)

            # build csr_matrix
            print('Building csr_matrix')
            fid_cid_spr_mat = csr_matrix((fid_cid_pair_cnt, (row_fid_cid, col_cid)), shape=(len(id2fid), len(id2cid)))
            fid_pid_spr_mat = csr_matrix((fid_pid_pair_cnt, (row_fid_pid, col_pid)), shape=(len(id2fid), len(id2pid)))
            print('Shape of fid_cid_mat: ', end=' ')
            print(fid_cid_spr_mat.shape)
            print('Shape of fid_pid_mat: ', end=' ')
            print(fid_pid_spr_mat.shape)
        
        # builfing nmf
        print('Building Non-Negative matrix factorization')
        # fid_cid_nmf = nimfa.Nmf(fid_cid_spr_mat, max_iter=MAX_ITER, rank=FID_CID_RANK, track_error=True, update='euclidean', objective='fro')
        # fid_pid_nmf = nimfa.Nmf(fid_pid_spr_mat, max_iter=MAX_ITER, rank=FID_PID_RANK, track_error=True, update='euclidean', objective='fro')
        fid_cid_nmf_mdl = NMF(n_components=FID_CID_RANK, init='random', random_state=0, max_iter=MAX_ITER, verbose=True )
        fid_pid_nmf_mdl = NMF(n_components=FID_PID_RANK, init='random', random_state=0, max_iter=MAX_ITER, verbose=True )
        
        # fitting to the data 
        print('Fitting to the data')
        # fid_cid_nmf_fit = fid_cid_nmf()
        # fid_pid_nmf_fit = fid_pid_nmf()
        
        
        # getting fid and cid W and H
        print('Getting fid and cid W and H')
        # fid_cid_W = fid_cid_nmf_fit.basis()
        # fid_cid_H = fid_cid_nmf_fit.coef()
        fid_cid_W = fid_cid_nmf_mdl.fit_transform(fid_cid_spr_mat)
        fid_cid_H = fid_cid_nmf_mdl.components_

        print('W shape: ', end=' ')
        print(fid_cid_W.shape)
        print('H shape: ', end=' ')
        print(fid_cid_H.shape)

        # getting fid and Pid W and H
        print('Getting fid and pid W and H')
        # fid_pid_W = fid_pid_nmf_fit.basis()
        # fid_pid_H = fid_cid_nmf_fit.coef()
        fid_pid_W = fid_pid_nmf_mdl.fit_transform(fid_pid_spr_mat)
        fid_pid_H = fid_pid_nmf_mdl.components_

        print('W shape: ', end=' ')
        print(fid_pid_W.shape)
        print('H shape: ', end=' ')
        print(fid_pid_H.shape)

        # split the feature to train and test
        # train_fid_cid_W =
        # test_fid_cid_W  

        # train_fid_pid_W
        # test_fid_pid_W
        # for 

        # writing feature data
        # writeCSV
        writePickle(fid_cid_spr_mat, path['MF_FID_CID_SPR_MAT_PKL_FILE'])
        writePickle(fid_pid_spr_mat, path['MF_FID_PID_SPR_MAT_PKL_FILE'])
        
        writePickle(fid_cid_W, path['MF_FID_CID_BASIS_PKL_FILE'])
        writePickle(fid_cid_H, path['MF_FID_CID_COEF_PKL_FILE'])
        writePickle(fid_pid_W, path['MF_FID_PID_BASIS_PKL_FILE'])
        writePickle(fid_pid_H, path['MF_FID_PID_COEF_PKL_FILE'])
        
        cid_feat = fid_cid_W
        pid_feat = fid_pid_W

    else:
        fid_cid_W = readPickle(path['MF_FID_CID_BASIS_PKL_FILE'])
        fid_pid_W = readPickle(path['MF_FID_PID_BASIS_PKL_FILE'])
        cid_feat = fid_cid_W
        pid_feat = fid_pid_W

    return cid_feat, pid_feat

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    cid_feat, pid_feat = getCidAndPidFeatureFromMF(True)
    end_time = datetime.datetime.now()
    print('Total time spent: ', end='')
    print(end_time - start_time)