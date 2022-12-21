# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 23:25:36 2022

@author: LENOVO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel("CocaCola_Sales_Rawdata.xlsx")
df
df.shape

df["first"]=df["Quarter"].str.split('_').str[0]
df["second"]=df["Quarter"].str.split('_').str[1]
df

# Getting dummies
df1=pd.get_dummies(df["second"])

# Creating t variable i.e. no of months
lst=list(range(1,43))
df2=pd.DataFrame(lst,columns=["t"])

# Square of t
lst=list(range(1,43))
df3=pd.DataFrame(lst,columns=["t_square"])
df4=df3**2.

X1=df["Sales"]
df["log_Sales"]=np.log(X1)
df

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
Y=LE.fit_transform(df["first"])
Y=pd.DataFrame(Y,columns=['year'])

# Concating
df=pd.concat([df,df1,df2,Y,df4],axis=1)
df
df.columns

# Plots
df["Sales"].plot()
df.boxplot(vert=False)
df.hist()

# Spliting
Train=df.head(34)
Test=df.tail(8)

# Models

# Linear Model 
import statsmodels.formula.api as smf
linear_model=smf.ols('Sales~year',data=Train).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(Test['year'])))
rmse_linear=np.sqrt(np.mean(np.array(Test['Sales'])-np.array(pred_linear))**2)
rmse_linear

# rmse=1847.942

# Exponential 
import statsmodels.formula.api as smf
EXP=smf.ols('log_Sales~year',data=Train).fit()
pred_EXP=pd.Series(EXP.predict(Test['year']))
rmse_EXP=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_EXP)))**2))
rmse_EXP

# rmse=1982.741

# Quadratic 
Quadratic = smf.ols("Sales~t+t_square",data=Train).fit()
pred_quad=pd.Series(Quadratic.predict(Test[["t","t_square"]]))
rmse_quad=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(pred_quad))**2))
rmse_quad

# rmse=437.745

# Additive seasonality
add_sea=smf.ols("Sales~86+87+88+89+90+91+92+93+94+95+96",data=Train).fit()
pred_add_sea=pd.Series(add_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea)**2)))
rmse_add_sea


# Additive seasonality Quadratic
add_sea_Quadratic=smf.ols("Sales~86+87+88+89+90+91+92+93+94+95+96",data=Train).fit()
pred_add_sea_Quadratic=pd.Series(add_sea_Quadratic.predict(Test[['86','87','88','89','90','91','92','93','94','95','96']]))
rmse_add_sea_Quadratic=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_Quadratic)**2)))
rmse_add_sea_Quadratic


# Multiplicative seasonability
Mul_sea=smf.ols("log_Sales~86+87+88+89+90+91+92+93+94+95+96",data=Train).fit()
pred_Mul_sea=pd.Series(Mul_sea.predict(Test))
rmse_Mul_sea=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mul_sea)))**2))
rmse_Mul_sea

# rmse=4320

# Multiplicative addictive seasonability
Mul_add_sea=smf.ols("log_Sales~86+87+88+89+90+91+92+93+94+95+96",data=Train).fit()
pred_Mul_add_sea=pd.Series(Mul_add_sea.predict(Test))
rmse_Mul_add_sea=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mul_sea)))**2))
rmse_Mul_add_sea

# rmse=4320


# Comparing the results
data ={"MODEL":pd.Series(["rmse_linear","rmse_EXP","rmse_quad","rmse_add_sea","rmse_add_sea_Quadratic","rmse_Mul_sea","rmse_Mul_add_sea"])}
data1 ={"MODEL":pd.Series([rmse_linear,rmse_EXP,rmse_quad,rmse_add_sea,rmse_add_sea_Quadratic,rmse_Mul_sea,rmse_Mul_add_sea])}

a=pd.DataFrame(data)
b=pd.DataFrame(data1)
new=pd.concat([a,b],axis=1)
new

# Best rmse value we are getting from Quadratic model i,e rmse_quad : 58.30476227446929


model_full =  smf.ols("Sales~t+t_square",data=df).fit()

df.dtypes
df.drop(df.columns[[0,3,1,4,17]],axis=1,inplace=True)
df.columns

pred_new =pd.Series(model_full .predict(df))
df["new_Passengers"]=pd.Series(pred_new)
df
df.columns
df["new_Passengers"]
