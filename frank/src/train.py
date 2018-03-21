import numpy as np
import pandas as pd
import argparse, os
import xgboost as xgb
from bayes_opt import BayesianOptimization

from utils import readCSV, getPath, writePickle
from logHandler import Logger

def prepare_data():
    
    datas = np.asarray(readCSV(args.train_feature_file))
    # print(np.array(datas).shape)
    # feat2idx = {}
    # idx2feat = {}
    # fid2idx = {}
    # idx2fid = {}
    # for idx, line in enumerate(datas):
    #     if idx == 0:
    #         feats = line.split(',')[1:]
    #         for feat_idx in range(len(feats)):
    #             feat2idx[feats[feat_idx]] = feat_idx
    #             idx2feat[feat_idx] = feats[feat_idx]
    #     else:
    #         fid = line.split(',', 1)[0]
    #         fid2idx[fid] = idx - 1
    #         idx2fid[idx - 1] = fid
    trainX = datas[1:, 1:]
    logger.info('Selected train features shape = ( %d , %d )', trainX.shape[0], trainX.shape[1])
    datas = np.asarray(readCSV(args.train_ans_file))
    trainY = datas[1:, 1:]
    logger.info('Selected train ans shape = ( %d )', trainY.shape[0])
    datas = np.asarray(readCSV(args.test_feature_file))
    testX = datas[1:, 1:]
    testFid = datas[1:, 0]
    logger.info('Selected test features shape = ( %d , %d )', testX.shape[0], testX.shape[1])
    return trainX, trainY, testX, testFid #, feat2idx, idx2feat, fid2idx, idx2fid

def str2float(X, Y, testX):
    X = X.astype(float)
    Y = Y.astype(float)
    testX = testX.astype(float)
    return X, Y, testX

def scale_data(X, testX):
    x_mean = np.mean(np.vstack([X, testX]), axis=0)
    x_std = np.std(np.vstack([X, testX]), axis=0) + (10**-8)
    
    X = X - np.tile(x_mean, (len(X), 1))
    X = X / np.tile(x_std, (len(X), 1))
    testX = testX - np.tile(x_mean, (len(testX), 1))
    testX = testX / np.tile(x_std, (len(testX), 1))
    return X, testX

def sample_data(X, Y):
    data_size = X.shape[0]
    # print(data_size)
    
    rand_idx = np.arange(data_size)
    np.random.shuffle(rand_idx)
    trainX = X[rand_idx[:int(data_size * (1 - args.valid_ratio))]]
    
    trainY = Y[rand_idx[:int(data_size * (1 - args.valid_ratio))]]
    validX = X[rand_idx[int(data_size * (1 - args.valid_ratio)):]]
    validY = Y[rand_idx[int(data_size * (1 - args.valid_ratio)):]] 
    logger.info('trainX shape = ( %5d, %3d )' % (trainX.shape[0], trainX.shape[1]))
    logger.info('trainY shape = ( %5d )' % (trainY.shape[0]))
    logger.info('validX shape = ( %5d, %3d )' % (validX.shape[0], validX.shape[1]))
    logger.info('validY shape = ( %5d )' % (validY.shape[0]))
    
    return trainX, trainY, validX, validY

def train_xgb(max_depth, subsample, min_child_weight, alpha, gamma, colsample_bytree, colsample_bylevel, learning_rate):
    xgb_params = {
        'nthread': args.nthread,
        'n_estimators': args.xgb_n_estimators,
        'eta': args.xgb_eta,
        'silent': args.xgb_silent,
        'seed': 0,
        # for _train_internal
        'eval_metric': [args.xgb_eval_metric],
        ######################
        'max_depth': int(max_depth),
        'subsample': max(min(subsample, 1), 0),
        'min_child_weight': int(min_child_weight),
        'alpha': max(alpha, 0),
        'gamma': max(gamma, 0),
        'colsample_bytree': max(min(colsample_bytree, 1), 0),
        'colsample_bylevel': max(min(colsample_bylevel, 1), 0),
        'learning_rate': max(min(learning_rate, 1), 0),
    }
    # print(dTrain)
    score = xgb.cv(
        xgb_params, 
        dTrain, 
        num_boost_round=args.xgb_num_boost_rounds, 
        early_stopping_rounds=args.xgb_early_stopping_rounds,
        verbose_eval=args.xgb_verbose_eval,
        maximize=args.xgb_maximize,
        nfold=args.xgb_nfold,
        seed=np.random.get_state()[1][0]
    )
    # logger.info(score)
    return score['test-auc-mean'].iloc[-1]

def train():
    X, Y, testX, testFid = prepare_data()
    X, Y, testX = str2float(X, Y, testX)
    X, testX = scale_data(X, testX)
    
    if args.select_feature:
        # select feature
        pass
    for iter in range(args.sample_num):
        np.random.seed(iter)
        logger.info('Iteration %2d, Current random seed: %2d' % (iter, np.random.get_state()[1][0]))
        # sampling data
        # trainX, trainY, validX, validY = sample_data(X, Y)
        trainX = X
        trainY = Y
        global dTrain
        dTrain = xgb.DMatrix(trainX, trainY, nthread=args.nthread)
        global dTest
        dTest = xgb.DMatrix(testX, nthread=args.nthread)

        # bayes_opt selection
        # parameter to be learning 
        logger.info('Setting parameters for BayesianOptimaization')       
        params = {
            'max_depth': (10, args.xgb_max_depth),
            'subsample': (args.xgb_subsample, 1),
            'min_child_weight': (1, args.xgb_min_child_weight),
            'alpha': (0, args.xgb_alpha),
            'gamma': (0, args.xgb_gamma),
            'colsample_bytree': (args.xgb_colsample_bytree, 1),
            'colsample_bylevel': (args.xgb_colsample_bylevel, 1),
            'learning_rate': (args.xgb_learning_rate_lower, args.xgb_learning_rate_upper)
        }
        logger.info('Running BayesianOptimization')
        xgb_bayesopt = BayesianOptimization(train_xgb, params)
        xgb_bayesopt.maximize(init_points=5, n_iter=25)
        
        # get the best param
        best_params = xgb_bayesopt.res['max']['max_params']
        logger.info('Iteration: %d, XGBoost max auc: %f' % (iter, xgb_bayesopt.res['max']['max_val']))
        for param, val in best_params.items():
            logger.info('Param %s: %r' % (param, val))
        # setting xgboost param
        logger.info('Setting best parameters for BayesianOptimization')
        xgb_params = {
            'nthread': args.nthread,
            'n_estimators': args.xgb_n_estimators,
            'eta': args.xgb_eta,
            'silent': args.xgb_silent,
            # for _train_internal
            'eval_metric': [args.xgb_eval_metric],
            ######################
            'max_depth': int(best_params['max_depth']),
            'subsample': max(min(best_params['subsample'], 1), 0),
            'min_child_weight': int(best_params['min_child_weight']),
            'alpha': max(best_params['alpha'], 0),
            'gamma': max(best_params['gamma'], 0),
            'colsample_bytree': max(min(best_params['colsample_bytree'], 1), 0),
            'colsample_bylevel': max(min(best_params['colsample_bylevel'], 1), 0),
            'learning_rate': max(min(best_params['learning_rate'], 1), 0),
        }
        # training
        model = xgb.train(
            xgb_params, 
            dTrain, 
            num_boost_round=args.xgb_num_boost_rounds, 
            verbose_eval=args.xgb_verbose_eval, 
            maximize=args.xgb_maximize
        )
        writePickle(model, os.path.join('mdl', 'model_iter%d_%dfold_%f.pkl' % (iter, args.xgb_nfold, xgb_bayesopt.res['max']['max_val'])))
        # predict valid y
        predY = model.predict(dTest)
        result_df = pd.DataFrame(data={'y':predY})
        joined_df = pd.DataFrame(testFid).join(result_df)


        joined_df.to_csv(os.path.join('result', 'xgb_result%d_%dfold.csv' % (iter, args.xgb_nfold)), index=False)


        

        # re-sorted the fid because of the random splitting data
        logger.info('----------------------------------------------------------------------\n\n\n')
if __name__ == '__main__':
    # setting file path
    path = getPath()
    # setting logger
    logger = Logger().getLogger()
    logger.info('======================== Execution of train.py ========================')    
    # setting args
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_num', type=int, default=10, help='the number of trying sampling')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='the ratio of data to be selected as validation')
    parser.add_argument('--select_feature', type=bool, default=False, help='decide whether to select feature')
    parser.add_argument('--select_feature_num', type=int, default=100, help='the feature number selected. if select_feature is False, this flag is useless')
    parser.add_argument('--train_feature_file', type=str, default=path['SELECTED_TRAIN_FEATURE_CSV_FILE'], help='the file of training features')
    parser.add_argument('--test_feature_file', type=str, default=path['SELECTED_TEST_FEATURE_CSV_FILE'], help='the file of testing features')
    parser.add_argument('--train_ans_file', type=str, default=path['TRAIN_ANS_CSV_FILE'], help='the file of training data ans')
    # xgboost usage args
    parser.add_argument('--nthread', type=int, default=36, help='number of working thread')
    parser.add_argument('--xgb_n_estimators', type=int, default=1000, help='xgboost n_estimators')
    parser.add_argument('--xgb_eta', type=float, default=0.01, help='xgboost eta')
    parser.add_argument('--xgb_eval_metric', type=str, default='auc', help='metric for evaluation')
    parser.add_argument('--xgb_max_depth', type=int, default=30, help='xgboost max_depth')
    parser.add_argument('--xgb_subsample', type=float, default=0.6, help='xgboost subsample')
    parser.add_argument('--xgb_silent', type=bool, default=True, help='xgboost silent')
    parser.add_argument('--xgb_min_child_weight', type=int, default=20, help='xgboost min_child_weight')
    parser.add_argument('--xgb_alpha', type=int, default=10, help='xgboost alpha')
    parser.add_argument('--xgb_gamma', type=int, default=10, help='xgboost gamma')
    parser.add_argument('--xgb_colsample_bytree', type=float, default=0.1, help='xgboost colsample_bytree')
    parser.add_argument('--xgb_colsample_bylevel', type=float, default=0.1, help='xgboost colsample_bylevel')
    parser.add_argument('--xgb_learning_rate_lower', type=float, default=0.01, help='lower bound of xgboost learning rate')
    parser.add_argument('--xgb_learning_rate_upper', type=float, default=0.1, help='upper bound of xgboost learning rate')
    
    parser.add_argument('--xgb_num_boost_rounds', type=int, default=1500, help='the number of boost round')
    parser.add_argument('--xgb_early_stopping_rounds', type=int, default=50, help='the rounds for early stop')
    parser.add_argument('--xgb_verbose_eval', type=bool, default=False, help='print the verbose eval or not')
    parser.add_argument('--xgb_maximize', type=bool, default=True, help='decide to maximize or minimize')
    parser.add_argument('--xgb_nfold', type=int, default=5, help='n fold for cv')
    args = parser.parse_args()
    for arg, value in sorted(vars(args).items()):
        logger.info('Argument %s: %r', arg, value)
    
    dTrain = []
    dTest = []

    train()
    logger.info('========================   End of Execution    ========================')
    