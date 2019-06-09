import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from operator import itemgetter
import xgboost as xgb
import random
import time
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc,recall_score,precision_score
import datetime as dt
from sklearn.model_selection import StratifiedKFold
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



    
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance

def get_features(train, test):
    trainval = list(train.columns.values)
    output = trainval
    return sorted(output)


def run_single(features, X_train, X_valid, y_train, y_valid, target, random_state=0):
    eta = 0.1
    max_depth= 6 
    subsample = 1
    colsample_bytree = 1
    min_chil_weight=1
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "min_chil_weight":min_chil_weight,
        "seed": random_state,
        #"num_class" : 22,
    }
    num_boost_round = 500
    early_stopping_rounds = 20
    test_size = 0.1

   
    
    
    dtrain = xgb.DMatrix(X_train[features], y_train, missing=-99) #
    dvalid = xgb.DMatrix(X_valid[features], y_valid, missing =-99) #

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration+1)
    
    #area under the precision-recall curve
    print(y_valid.values)
    score = average_precision_score(y_valid.values, check)
    print('area under the precision-recall curve: {:.6f}'.format(score))

    
    check2=check.round()
    score = precision_score(y_valid.values, check2)
    print('precision score: {:.6f}'.format(score))

    score = recall_score(y_valid.values, check2)
    print('recall score: {:.6f}'.format(score))
    
    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set... ")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration+1)
    score = average_precision_score(y_valid.values, test_prediction)

    print('area under the precision-recall curve test set: {:.6f}'.format(score))
    
    ############################################ ROC Curve
    

 
    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_valid.values, check)
    roc_auc = auc(fpr, tpr)
    #xgb.plot_importance(gbm)
    #plt.show()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    ##################################################


    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction, imp, gbm.best_iteration+1


#load data from csv file
# Any results you write to the current directory are saved as output.
start_time = dt.datetime.now()
print("Start time: ",start_time)
    
Spectrum=pd.read_csv('Spectrum_baseline.csv')
Spectrum.fillna(method='ffill', inplace = True)
Spectrum.fillna(value=0.0, inplace = True)
feature = Spectrum.columns
target = pd.read_csv('Spectrum_labels_binary.csv')
data = Spectrum[feature[1:]]
ind1 = target.index[target['labels'] ==1].tolist()
ind0 = target.index[target['labels'] ==0].tolist()
train_0 = data.loc[ind0]
train_1 = data.loc[ind1]
y_0 = target.loc[ind0]
y_1 = target.loc[ind1]
#test-train split
number_train0 = int(len(train_0)*0.8)
number_train1 = int(len(train_1)*0.8)
train = pd.concat([train_0[:number_train0],train_1[:number_train1]],axis=0)
test = pd.concat([train_0[number_train0:],train_1[number_train1:]],axis=0)
y_train = pd.concat([y_0[:number_train0],y_1[:number_train1]],axis=0)
y_test = pd.concat([y_0[number_train0:],y_1[number_train1:]],axis=0)
print('train shape',train.shape, y_train.shape)
print('test shape',test.shape, y_test.shape)
print("Building model.. ",dt.datetime.now()-start_time)
features = list(train.columns.values)
preds, imp, num_boost_rounds = run_single(features[:-1],train, test,y_train, y_test, 'labels',42)