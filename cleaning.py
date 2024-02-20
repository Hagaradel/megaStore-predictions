import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import metrics

def data_clean(data):


    data['Ship Date'].fillna('0')
    data['Order Date'].fillna('0')
    data['Ship Date']=pd.to_datetime(data['Ship Date'],format="%m/%d/%Y")
    data['Order Date']=pd.to_datetime(data['Order Date'],format="%m/%d/%Y")
    duration=(data['Ship Date']-data['Order Date']).dt.days
    data['Duration']=duration
    data=data.drop(columns=['Country','State','Region','Product ID','Row ID','Order Date','Ship Date','Order ID'])

#################################################################
    r=data['CategoryTree'].str.split(',', expand=True)
    t=r[0].str.split(':',expand=True)
    r2=r[1].str.split('}',expand=True)
    r3=r2[0].str.split(':',expand=True)
##########################################sub
    SubCategory=r3[1].str.split("'",expand=True)
    yk=t[1].str.split("{",expand=True)
######################tech
    MainCategory=yk[0].str.split("'",expand=True)

    data['Main Category']=MainCategory[1]
    data['Sub Category']=SubCategory[1]

    data=data.drop(columns=['CategoryTree'])
#####################################################
    data['Discount']=data['Discount']*100
    return data
