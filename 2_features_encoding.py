# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

############################### DATA IMPORT FROM FILE ##########################
data=pd.read_csv("D:/current/Python/my_exercise/wta_data.csv",low_memory=False)
data.Date = data.Date.apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

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

features.to_csv("D:/current/Python/my_exercise/wta_data_features.csv",index=False)