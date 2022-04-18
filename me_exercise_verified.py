# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:06:11 2021

@author: Lab5
"""

import pandas as pd
import numpy as np
from datetime import datetime,timedelta

import glob
filenames=list(glob.glob("C:/Users/sbsm1/Documents/my_exercise/20*.xls*"))
l = [pd.read_excel(filename) for filename in filenames]
no_b365=[i for i,d in enumerate(l) if "B365W" not in l[i].columns]
no_pi=[i for i,d in enumerate(l) if "PSW" not in l[i].columns]
for i in no_pi:
    l[i]["PSW"]=np.nan
    l[i]["PSL"]=np.nan
for i in no_b365:
    l[i]["B365W"]=np.nan
    l[i]["B365L"]=np.nan
l=[d[list(d.columns)[:13]+["Wsets","Lsets","Comment"]+["PSW","PSL","B365W","B365L"]] for d in [l[0]]+l[2:]]
data=pd.concat(l,0)

### Data cleaning
data=data.sort_values("Date")
data["WRank"]=data["WRank"].replace(np.nan,0)
data["WRank"]=data["WRank"].replace("NR",2000)
data["LRank"]=data["LRank"].replace(np.nan,0)
data["LRank"]=data["LRank"].replace("NR",2000)
data["WRank"]=data["WRank"].astype(int)
data["LRank"]=data["LRank"].astype(int)
data["Wsets"]=data["Wsets"].astype(float)
data["Lsets"]=data["Lsets"].replace("`1",1)
data["Lsets"]=data["Lsets"].astype(float)
data = data[data["Comment"] == "Completed"] #здесь оставляем только завершенные матчи
#data = data.dropna() #здесь выбрасываем все строки где есть пустые поля
data=data.reset_index(drop=True)

### Storage of the raw dataset
data.to_csv("C:/Users/sbsm1/Documents/my_exercise/wta_data.csv",index=False)

data=pd.read_csv("C:/Users/sbsm1/Documents/my_exercise/wta_data.csv",low_memory=False)
data.Date = data.Date.apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

############################### CATEGORICAL FEATURES ENCODING ##################

### The features "player1", "player2" and "Tournament" are treated differently
### from the other features. 

def categorical_features_encoding(cat_features):
    """
    Categorical features encoding.
    Simple one-hot encoding.
    """
    cat_features=cat_features.apply(preprocessing.LabelEncoder().fit_transform)
    ohe=OneHotEncoder()
    cat_features=ohe.fit_transform(cat_features)
    cat_features=pd.DataFrame(cat_features.todense())
    cat_features.columns=["cat_feature_"+str(i) for i in range(len(cat_features.columns))]
    cat_features=cat_features.astype(int)
    return cat_features

def features_players_encoding(data):
    """
    Encoding of the players . 
    The players are not encoded like the other categorical features because for each
    match we encode both players at the same time (we put a 1 in each row corresponding 
    to the players playing the match for each match).
    """
    winners=data.Winner
    losers=data.Loser
    le = preprocessing.LabelEncoder()
    le.fit(list(winners)+list(losers))
    winners=le.transform(winners)
    losers=le.transform(losers)
    encod=np.zeros([len(winners),len(le.classes_)])
    for i in range(len(winners)):
        encod[i,winners[i]]+=1
    for i in range(len(losers)):
        encod[i,losers[i]]+=1
    columns=["player_"+el for el in le.classes_]
    players_encoded=pd.DataFrame(encod,columns=columns)
    return players_encoded

def features_tournaments_encoding(data):
    """
    Encoding of the tournaments . 
    """
    tournaments=data.Tournament
    le = preprocessing.LabelEncoder()
    tournaments=le.fit_transform(tournaments)
    encod=np.zeros([len(tournaments),len(le.classes_)])
    for i in range(len(tournaments)):
        encod[i,tournaments[i]]+=1
    columns=["tournament_"+el for el in le.classes_]
    tournaments_encoded=pd.DataFrame(encod,columns=columns)
    return tournaments_encoded

########################### Selection of our period ############################

odds = data[["PSW","PSL"]]

########################## Encoding of categorical features ####################

features_categorical = data[["Tier","Court","Surface","Round","Tournament"]]
features_categorical_encoded = categorical_features_encoding(features_categorical)
players_encoded = features_players_encoding(data)
tournaments_encoded = features_tournaments_encoding(data)
features_onehot = pd.concat([features_categorical_encoded,players_encoded,tournaments_encoded],1)

# Categorical features
features_onehot = pd.DataFrame(np.repeat(features_onehot.values,2, axis=0),columns=features_onehot.columns)

# odds feature
features_odds = pd.Series(odds.values.flatten(),name="odds")
features_odds = pd.DataFrame(features_odds)

### Building of the final dataset
# You can remove some features to see the effect on the ROI
features = pd.concat([features_odds, features_onehot],1)

features.to_csv("C:/Users/sbsm1/Documents/my_exercise/wta_data_features.csv",index=False)

######################### Confidence computing for each match ############################
features=pd.read_csv("C:/Users/sbsm1/Documents/my_exercise/wta_data_features.csv")

start_date=data.Date[0] #first day of testing set
test_beginning_match=data[data.Date==start_date].index[0] #id of the first match of the testing set
span_matches=len(data)-test_beginning_match+1
duration_val_matches=100 
duration_train_matches=int(len(data)*0.7)
duration_test_matches=int(len(data)*0.3)

### XGBoost model
import xgboost as xgb
def xgbModelBinary(xtrain,ytrain,xval,yval,p,sample_weights=None):
    """
    XGB model training. 
    Early stopping is performed using xval and yval (validation set).
    Outputs the trained model, and the prediction on the validation set
    """
    if sample_weights==None:
        dtrain=xgb.DMatrix(xtrain,label=ytrain)
    else:
        dtrain=xgb.DMatrix(xtrain,label=ytrain,weight=sample_weights)
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
    end_val=min(beg_test-1,nm-1)
    beg_val=beg_test-duration_val_matches
    end_train=beg_val-1
    beg_train=beg_val-duration_train_matches
       
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
    model=xgbModelBinary(xtrain,ytrain,xval,yval,xgb_params,sample_weights=None)
    
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

##TODO: переписать функцию которая предсказывает конфиденс
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
conf.to_csv("C:/Users/sbsm1/Documents/my_exercise/confidence_data.csv",index=False)

##########################################################################################################
############################## PROFIT COMPUTATION AND VISUALISATION ######################################
##########################################################################################################
import matplotlib.pyplot as plt

conf = pd.read_csv("C:/Users/sbsm1/Documents/my_exercise/confidence_data.csv",low_memory=False) #загружаем еще раз чтобы не пачкать
# lengthconfidence = len(conf)
# conf.iloc[[0]]
conf=conf.dropna() #выбрасываем строки, где есть пустые ячейки
conf=conf.sort_values("date",ascending=True) #сортируем по дате
conf=conf.reset_index(drop=True) #обновляем индекс

stavka=1 #размер ставки
p=100 #начальная сумма

profit_all = []
for i in range(0,len(conf)):
    if conf.loc[i,'win0'] == 1:
        p = p + stavka*conf.loc[i,'PSW']
        profit_all.append(round(p,2))
    else:
        p = p - stavka*conf.loc[i,'PSW']
        profit_all.append(round(p,2))
#print(profit_all)

    # plt.plot(ticks,profit)
    # plt.xticks(range(0,101,5))

    # plt.suptitle(title)

ticks = range(0,len(conf))
plt.plot(ticks,profit_all)
plt.xlabel("Number of matches we bet on")
plt.ylabel("Bankroll")