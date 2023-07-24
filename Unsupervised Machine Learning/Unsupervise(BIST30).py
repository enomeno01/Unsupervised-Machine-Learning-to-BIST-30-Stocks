# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 18:50:06 2023

@author: enesd
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans,AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy.cluster.hierarchy import linkage,dendrogram

url="https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx?endeks=03#page-1"
r=requests.get(url)
s=BeautifulSoup(r.text,"html.parser")

tablo=s.find("table",{"id":"summaryBasicData"})
tablo=pd.read_html(str(tablo),flavor="bs4")[0]
print(tablo)

hisseler=[]

for i in tablo["Kod"]:
    hisseler.append(i)
    
parametreler=(
    ("hisse",hisseler[0]),
    ("startdate","23-07-2021"),
    ("enddate","23-07-2023")
    )

url2="https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil?"
r2=requests.get(url2,params=parametreler).json()["value"]
veri=pd.DataFrame.from_dict(r2)
veri=veri.iloc[:,0:3]
veri=veri.rename({"HGDG_HS_KODU":"Hisse","HGDG_TARIH":"Tarih","HGDG_KAPANIS":"Fiyat"},axis=1)

data={"Tarih":veri["Tarih"],veri["Hisse"][0]:veri["Fiyat"]}
veri=pd.DataFrame(data)

tumveri=[veri]

del hisseler[0]

for j in hisseler:
    parametreler=(
        ("hisse",j),
        ("startdate","23-07-2021"),
        ("enddate","23-07-2023")
        )

    url2="https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil?"
    r2=requests.get(url2,params=parametreler).json()["value"]
    veri=pd.DataFrame.from_dict(r2)
    veri=veri.iloc[:,0:3]
    veri=veri.rename({"HGDG_HS_KODU":"Hisse","HGDG_TARIH":"Tarih","HGDG_KAPANIS":"Fiyat"},axis=1)

    data={"Tarih":veri["Tarih"],veri["Hisse"][0]:veri["Fiyat"]}
    veri=pd.DataFrame(data)
    tumveri.append(veri)
    
df=tumveri[0]    

for son in tumveri[1:]:
    df=df.merge(son,on="Tarih")
    

veri=df.drop(columns="Tarih",axis=1)

gelir=veri.pct_change().mean()*252
sonuc=pd.DataFrame(gelir)
sonuc.columns=["Gelir"]

sonuc["Oynaklık"]=veri.pct_change().std()*np.sqrt(252)
sonuc=sonuc.reset_index()
sonuc=sonuc.rename({"index":"Hisse"},axis=1)

ms = MinMaxScaler()
X= ms.fit_transform(sonuc.iloc[:,[1,2]])
X= pd.DataFrame(X,columns=["Gelir","Oynaklık"])

    
kmodel=KMeans(random_state=(0))
grafik=KElbowVisualizer(kmodel, k=(2,20))
grafik.fit(X) 
grafik.poof()   


kmodel=KMeans(n_clusters=6,random_state=(0))
kfit=kmodel.fit(X)
labels=kfit.labels_

sbn.scatterplot(x="Gelir",y="Oynaklık",data=X,hue=labels,palette="deep")
plt.show()    
   

sonuc["Labels"]=labels

sbn.scatterplot(x="Labels",y="Hisse",data=sonuc,hue=labels,palette="deep")
plt.show()  
    

hc=linkage(X,method="single")
dendrogram(hc)
plt.show()

model=AgglomerativeClustering(n_clusters=4,linkage="single")
tahmin=model.fit_predict(X)
labels=model.labels_

sonuc["Labels"]=labels

sbn.scatterplot(x="Labels",y="Hisse",data=sonuc,hue="Labels",palette="deep")
plt.show()





















    