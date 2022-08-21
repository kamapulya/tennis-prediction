# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob

filenames=list(glob.glob("D:/current/Python/my_exercise/20*.xls*"))
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
#df = df.drop(['column_nameA', 'column_nameB'], axis=1)
data = data.drop(["Best of","B365W","B365L"], axis=1)
#data = data[data["Comment"] == "Completed"] #здесь оставляем только завершенные матчи ВОЗМОЖНО ТАК ДЕЛАТЬ НЕ СТОИТ!!!!
#data = data.dropna() #здесь выбрасываем все строки где есть пустые поля ВОЗМОЖНО НА ЭТОМ ЭТАПЕ НЕ НУЖНО
data=data.reset_index(drop=True)

### Storage of the raw dataset
data.to_csv("D:/current/Python/my_exercise/wta_data.csv",index=False)