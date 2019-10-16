#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading datasets

# In[2]:


dataTrain = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
dataTest  = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")


# In[3]:


sns.pairplot(dataTrain, palette="husl")


# In[4]:


data = dataTrain
data.corr()


# # Removing least correlated columns and scaling target variable

# In[5]:


del_col_list = ['Instance','Hair Color', 'Wears Glasses']
data = data.drop(del_col_list, axis=1)
data= data[data["Income"]>0]
data.info()
data["Income"] = np.log(data["Income"])

data_test_original = dataTest
del_col_list = ['Instance','Hair Color', 'Wears Glasses']
dataTest = dataTest.drop(del_col_list, axis=1)


# # Filling NA values in columns with continuos value

# In[6]:


data[["Year of Record"]] = data[["Year of Record"]].fillna(value=data["Year of Record"].mode()[0])
data[["Age"]] = data[["Age"]].fillna(value=data["Age"].mean())

dataTest[["Year of Record"]] = dataTest[["Year of Record"]].fillna(value=dataTest["Year of Record"].mode()[0])
dataTest[["Age"]] = dataTest[["Age"]].fillna(value=dataTest["Age"].mean())


# # Scaling size of city

# In[7]:


year_max_value = data['Size of City'].max()
year_min_value = data['Size of City'].min()
data['Size of City'] = (data['Size of City'] - year_min_value) / (year_max_value - year_min_value)

year_max_value = dataTest['Size of City'].max()
year_min_value = dataTest['Size of City'].min()
dataTest['Size of City'] = (dataTest['Size of City'] - year_min_value) / (year_max_value - year_min_value)


# # Filling NA values in columns with discrete value and replacing absurd values

# In[8]:


data[["Gender"]] = data[["Gender"]].fillna(value="unknownG")
data[['Gender']] = data[['Gender']].replace('0', 'unknownG') 
data[['Gender']] = data[['Gender']].replace('unknown', 'unknownG')

data[["University Degree"]] = data[["University Degree"]].fillna(data["University Degree"].mode()[0])

data[["Profession"]] = data[["Profession"]].fillna(value="unknownP")


dataTest[["University Degree"]] = dataTest[["University Degree"]].fillna(dataTest["University Degree"].mode()[0])

dataTest[["Profession"]] = dataTest[["Profession"]].fillna(value="unknownP")

dataTest[["Gender"]] = dataTest[["Gender"]].fillna(value="unknownG")
dataTest[['Gender']] = dataTest[['Gender']].replace('0', 'unknownG') 
dataTest[['Gender']] = dataTest[['Gender']].replace('unknown', 'unknownG')


# # Converting discrete columns to continuos values 
# ## Used mean of Income for a given column index

# In[9]:


groupedProf = data.groupby('Profession', as_index=False)['Income'].mean()
data['Profession'] = data['Profession'].map(groupedProf.set_index('Profession')['Income'])

groupedCountry = data.groupby('Country', as_index=False)['Income'].mean()
data['Country'] = data['Country'].map(groupedCountry.set_index('Country')['Income'])

groupedUD = data.groupby('University Degree', as_index=False)['Income'].mean()
data['University Degree'] = data['University Degree'].map(groupedUD.set_index('University Degree')['Income'])

groupedG = data.groupby('Gender', as_index=False)['Income'].mean()
data['Gender'] = data['Gender'].map(groupedG.set_index('Gender')['Income'])


dataTest['Profession'] = dataTest['Profession'].map(groupedProf.set_index('Profession')['Income'])
dataTest['Country'] = dataTest['Country'].map(groupedCountry.set_index('Country')['Income'])
dataTest['University Degree'] = dataTest['University Degree'].map(groupedUD.set_index('University Degree')['Income'])
dataTest['Gender'] = dataTest['Gender'].map(groupedG.set_index('Gender')['Income'])


dataTest[["Country"]] = dataTest[["Country"]].fillna(value=dataTest["Country"].mean())
dataTest[["Profession"]] = dataTest[["Profession"]].fillna(value=dataTest["Profession"].mean())


# # Splitting data for training and testing

# In[10]:


X = data.drop('Income', axis = 1).values
y = data['Income'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=2)


# # Testing model and getting local RMSE

# In[11]:



model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

y_pred = y_pred.reshape(-1)

print("RMSE: " + str(sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred)))))


# # Equating number of columns and filling NA in test data

# In[12]:


missing_cols = set( data.columns ) - set( dataTest.columns )
for c in missing_cols:
    dataTest[c] = 0
dataTest = dataTest[data.columns]


missing_cols = set( dataTest.columns ) - set( data.columns )
for c in missing_cols:
    data[c] = 0
data = data[dataTest.columns]

del_col_list = ['Income']
dataTest = dataTest.drop(del_col_list, axis=1)


# # Final prediction

# In[13]:


X = data.drop('Income', axis = 1).values
y = data['Income'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)


model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(dataTest)
y_pred = y_pred.reshape(-1)


# # Writing to CSV

# In[14]:


instances = data_test_original['Instance'].to_numpy()
to_print = pd.DataFrame({'Instance': instances, 'Income': np.exp(y_pred)})
to_print.to_csv('result.csv', index=False)

