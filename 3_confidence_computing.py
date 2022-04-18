# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 14:46:57 2021

@author: Lab5
"""

import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb


data=pd.read_csv("D:/current/Python/my_exercise/wta_data.csv",low_memory=False)
data.Date = data.Date.apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))


######################### Confidence computing for each match ############################
features=pd.read_csv("D:/current/Python/my_exercise/wta_data_features.csv")
#TODO: Переписать так, чтобы начало тестового сета определялось либо в процентах от выборки, либо от даты

start_date=data.Date[1750] #first day of testing set
test_beginning_match=data[data.Date==start_date].index[0] #id of the first match of the testing set
span_matches=len(data)-test_beginning_match+1
duration_test_matches=span_matches
duration_val_matches=100 
duration_train_matches=int(len(data) - duration_test_matches - duration_val_matches)


### XGBoost model
def xgbModelBinary(xtrain,ytrain,xval,yval,p):
    """
    XGB model training. 
    Early stopping is performed using xval and yval (validation set).
    Outputs the trained model, and the prediction on the validation set
    """
    dtrain=xgb.DMatrix(xtrain,label=ytrain)
    dval=xgb.DMatrix(xval,label=yval)
    eval_set = [(dtrain,"train_loss"),(dval, 'eval')]
    params={'eval_metric':"logloss","objective":"binary:logistic",'subsample':0.8,
            'min_child_weight':p[2],'alpha':p[6],'lambda':p[5],'max_depth':int(p[1]),
            'gamma':p[3],'eta':p[0],'colsample_bytree':p[4]}
    model=xgb.train(params, dtrain, int(p[7]),evals=eval_set,early_stopping_rounds=int(p[8]))
    return model

## XGB parameters
learning_rate=[0.295] 
max_depth=[19]
min_child_weight=[1]
gamma=[0.8]
csbt=[0.5]
lambd=[0]
alpha=[2]
num_rounds=[300]
early_stop=[5]
params=np.array(np.meshgrid(learning_rate,max_depth,min_child_weight,gamma,csbt,lambd,alpha,num_rounds,early_stop)).T.reshape(-1,9).astype(np.float)
xgb_params=params[0]

def assessStrategyGlobal(test_beginning_match,
                         duration_train_matches,
                         duration_val_matches,
                         duration_test_matches,
                         xgb_params,
                         features,
                         data,
                         model_name="0"):
    """
    Given the id of the first match of the testing set (id=index in the dataframe "data"),
    outputs the confidence dataframe.
    The confidence dataframe tells for each match is our prediction is right, and for
    the outcome we chose, the confidence level.
    The confidence level is simply the probability we predicted divided by the probability
    implied by the bookmaker (=1/odd).
    """
    ########## Training/validation/testing set generation
    
    # Number of matches in our dataset (ie. nb. of outcomes divided by 2)
    nm=int(len(features)/2)
    
    # Id of the first and last match of the testing,validation,training set
    beg_test=test_beginning_match
    end_test=min(test_beginning_match+duration_test_matches-1,nm-1)
    
    beg_val=beg_test-duration_val_matches               ###начало валидационной выборки
    end_val=min(beg_test-1,nm-1)                        ###конец валидационной выборки
    
    beg_train=beg_val-duration_train_matches            ###начало тренировочной выборки
    end_train=beg_val-1                                 ###конец тренировочной выборки
       
    train_indices=range(2*beg_train,2*end_train+2)
    val_indices=range(2*beg_val,2*end_val+2)
    test_indices=range(2*beg_test,2*end_test+2)
    
    if (len(test_indices)==0)|(len(train_indices)==0):
        return 0
    
    # Split in train/validation/test
    xval=features.iloc[val_indices,:].reset_index(drop=True)
    xtest=features.iloc[test_indices,:].reset_index(drop=True)
    xtrain=features.iloc[train_indices,:].reset_index(drop=True)
    ytrain=pd.Series([1,0]*int(len(train_indices)/2))
    yval=pd.Series([1,0]*int(len(val_indices)/2))
       
    ### ML model training
    model=xgbModelBinary(xtrain,ytrain,xval,yval,xgb_params)
    
    # The probability given by the model to each outcome of each match :
    pred_test= model.predict(xgb.DMatrix(xtest,label=None)) 
    # For each match, the winning probability the model gave to the players that won (should be high...) :
    prediction_test_winner=pred_test[range(0,len(pred_test),2)]
    # For each match, the winning probability the model gave to the players that lost (should be low...) :
    prediction_test_loser=pred_test[range(1,len(pred_test),2)]
    
    ### Odds and predicted probabilities for the testing set (1 row/match)
    odds=data[["PSW","PSL"]].iloc[range(beg_test,end_test+1)]
    implied_probabilities=1/odds
    p=pd.Series(list(zip(prediction_test_winner,prediction_test_loser,implied_probabilities.PSW,implied_probabilities.PSL)))

    ### For each match in the testing set, if the model predicted the right winner :
    right=(prediction_test_winner>prediction_test_loser).astype(int)
    ### For each match in the testing set, the confidence of the model in the outcome it chose
    def sel_match_confidence(x):
        if x[0]>x[1]:
            return x[0]/x[2] 
        else:
            return x[1]/x[3] 
    confidence=p.apply(lambda x:sel_match_confidence(x))
    
    ### The final confidence dataset 
    confidenceTest=pd.DataFrame({"match":range(beg_test,end_test+1),
                                 "win"+model_name:right,
                                 "confidence"+model_name:confidence,
                                 "PSW":odds.PSW.values})
    confidenceTest=confidenceTest.sort_values("confidence"+model_name,ascending=False).reset_index(drop=True)
    
    return confidenceTest

# We predict the confidence in each outcome, "duration_test_matches" matches at each iteration
key_matches=np.array([test_beginning_match+duration_test_matches*i for i in range(int(span_matches/duration_test_matches)+1)])
confs=[]
for start in key_matches:
    conf=assessStrategyGlobal(start,duration_train_matches,duration_val_matches,duration_test_matches,xgb_params,features,data)
    confs.append(conf)
confs=[el for el in confs if type(el)!=int]
conf=pd.concat(confs,0)

## We add the date to the confidence dataset (can be useful for analysis later)
dates=data.Date.reset_index()
dates.columns=["match","date"]
conf=conf.merge(dates,on="match")
conf=conf.sort_values("confidence0",ascending=False)
conf=conf.reset_index(drop=True)

# We store this dataset
conf.to_csv("D:/current/Python/my_exercise/confidence_data.csv",index=False)

conf_merged=data.merge(conf, left_index=True, right_on='match')
conf_merged.to_csv("D:/current/Python/my_exercise/confidence_data_merged.csv",index=False)