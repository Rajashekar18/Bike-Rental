#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import pandas as pd
import seaborn as sns
from random import randrange, uniform
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics  

# In[11]:

os.chdir("C:/Users/Rajashekar/Videos/python_project")
os.getcwd()

# In[12]:
bike_data=pd.read_csv("day.csv",sep=",")

# In[13]:

bike_data.columns


# In[14]:

#these are features for anova test
cnames_anova=['season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit','cnt']
#below features for correlation analysis
cnames_corr=['temp','atemp','hum','windspeed','cnt']


# In[15]:
#saving into new dataset called bike_data_1
bike_data_1=bike_data.loc[:,cnames_corr]
bike_data_1.shape


# In[16]:
#below is correlation matrix
f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corr = bike_data_1.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[17]:
#Anova for categorical variables
import statsmodels.api as sm
from statsmodels.formula.api import ols
bike_data_2=bike_data.loc[:,cnames_anova]

#bike_data_2.boxplot('cnt',by='season')
season_anova=ols('cnt~season',data=bike_data_2).fit()
season_anova_table=sm.stats.anova_lm(season_anova,type=2)
print(season_anova_table)

yr_anova=ols('cnt~yr',data=bike_data_2).fit()
yr_anova_table=sm.stats.anova_lm(yr_anova,type=2)
print(yr_anova_table)

month_anova=ols('cnt~mnth',data=bike_data_2).fit()
month_anova_table=sm.stats.anova_lm(month_anova,type=2)
print(month_anova_table)

weekday_anova=ols('cnt~weekday',data=bike_data_2).fit()
weekday_anova_table=sm.stats.anova_lm(weekday_anova,type=2)
print(weekday_anova_table)

workingday_anova=ols('cnt~workingday',data=bike_data_2).fit()
workingday_anova_table=sm.stats.anova_lm(workingday_anova,type=2)
print(workingday_anova_table)

weathersit_anova=ols('cnt~weathersit',data=bike_data_2).fit()
weathersit_anova_table=sm.stats.anova_lm(weathersit_anova,type=2)
print(weathersit_anova_table)

holiday_anova=ols('cnt~holiday',data=bike_data_2).fit()
holiday_anova_table=sm.stats.anova_lm(holiday_anova,type=2)
print(holiday_anova_table)


# In[18]:

#below is the step to remove columns that contain hign +ve and -ve correlations and also categorical features that gives no 
#information to the target variable
bike_data_final=bike_data.drop(['instant','dteday','weekday','workingday','holiday','atemp','casual','registered'],axis=1)
x=bike_data_final.drop('cnt',axis=1)
y=bike_data_final['cnt']

#is to check column names of removing target feature
bike_data_final.columns

#below is the steps to divide dataset into train and test data sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#Applying DT regression
fit_Dt=DecisionTreeRegressor(max_depth=50).fit(x_train,y_train)


#storing predicted values into y_pred using above model called fit_Dt
y_pred=fit_Dt.predict(x_test)


#to see Actual values and predicted values
df=pd.DataFrame({'Actual':y_test,'Prediction':y_pred})
df

#calculation of MAPE
def MAPE(x,y):
    mape=np.mean(np.abs((x-y)/x))*100
    return mape

#Percentage of error in out DT Regression Model
count=MAPE(y_test,y_pred)
count

#below is to divide dataset into train and test data sets for linear regression
bike_data_final=bike_data.drop(['instant','dteday','weekday','workingday','holiday','atemp','casual','registered'],axis=1)
x=bike_data_final.drop('cnt',axis=1)
y=bike_data_final['cnt']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# Train the model using the linear regression 
model = sm.OLS(y_train, x_train).fit()

#summary of above model
model.summary()

#prediction of values of above model
y_pred= model.predict(x_test) 

#to see Actual values and predicted values
df=pd.DataFrame({'Actual':y_test,'Prediction':y_pred})

#MAPE1 function
def MAPE1(x,y):
    mape=np.mean(np.abs((x-y)/x))*100
    return mape
#Error percentage
count=MAPE1(y_test,y_pred)
count


#below is to divide dataset into train and test data sets for RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
bike_data=pd.read_csv("day.csv",sep=",")
bike_data_final=bike_data.drop(['instant','dteday','weekday','workingday','holiday','atemp','casual','registered'],axis=1)
x=bike_data_final.drop('cnt',axis=1)
y=bike_data_final['cnt']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#RandomForest Regressor
model_RF=RandomForestRegressor(n_estimators= 1000, random_state=0).fit(x_train,y_train)
y_pred= model.predict(x_test)
count=MAPE1(y_test,y_pred)
count

#prediction of values of above model
y_pred= model.predict(x_test) 

#to see Actual values and predicted values
df_RF=pd.DataFrame({'Actual':y_test,'Prediction':y_pred})

#Error percentage
count=MAPE1(y_test,y_pred)
count

