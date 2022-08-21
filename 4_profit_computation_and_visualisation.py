# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


data=pd.read_csv("D:/current/Python/my_exercise/wta_data.csv",low_memory=False)
data.Date = data.Date.apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

##########################################################################################################
############################## PROFIT COMPUTATION AND VISUALISATION ######################################
##########################################################################################################
conf = pd.read_csv("D:/current/Python/my_exercise/confidence_data.csv",low_memory=False) #загружаем еще раз чтобы не пачкать
# lengthconfidence = len(conf)
# conf.iloc[[0]]
conf=conf.dropna()
conf=conf.sort_values("date",ascending=True)
conf=conf.reset_index(drop=True)

stavka=10 #size of the bet
p=100 #initial sum
porog=0.1

profit_all = []
for i in range(0,len(conf)):
    if conf.loc[i,'confidence0'] > porog:
        if conf.loc[i,'win0'] == 1:
            p = p - stavka + stavka*conf.loc[i,'PSW']
            profit_all.append(round(p,2))
        else:
            p = p - stavka
            profit_all.append(round(p,2))
    
#print(profit_all)
print(len(profit_all))

ticks = range(0,len(profit_all))
plt.plot(ticks,profit_all)
plt.xlabel("Number of matches we bet on")
plt.ylabel("Bankroll")

# ### Comparison with the strategy "Always bet on the lowest odd"
# stavka2=1
# p2=100
# period = 1800
# profit_basic=[]
# for i in range(0,period):
#     if data.loc[i,'PSW'] < data.loc[i,'PSL']:
#         p2 = p2 - stavka2 + stavka2*data.loc[i,'PSW']
#         profit_basic.append(round(p2,2))
#     else:
#         p2 = p2 - stavka2
#         profit_basic.append(round(p2,2))
# #print(profit_basic)

# ticks = range(0,period)
# plt.plot(ticks,profit_basic)
# plt.xlabel("Number of matches we bet on")
# plt.ylabel("Bankroll")