#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


# In[45]:


np.set_printoptions(suppress=True)
pd.set_option('display.max_columns',8000)
pd.set_option('display.max_rows',7000)


# In[46]:


train=pd.read_csv('C:/Users/user/Desktop/IVY WORK BOOK/CASP.csv')


# In[47]:


train.head()


# In[48]:


train.isnull().sum()


# In[49]:


pd.plotting.scatter_matrix(train,figsize=(15,10))


# In[50]:


train.columns


# In[51]:


Predictors=['RMSD', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']

Target=['F9']

X=train[Predictors].values
y=train[Target].values


# In[52]:


from sklearn.preprocessing import StandardScaler
PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()


# In[53]:


X=PredictorScaler.fit_transform(X)
y=TargetVarScaler.fit_transform(y)


# In[54]:


X[:]


# In[55]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[56]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[58]:


from keras.models import Sequential
from keras.layers import Dense


# In[92]:


model = Sequential()
model.add(Dense(units=500, input_dim=9, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=100, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(units=40, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=1, kernel_initializer='normal', activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[93]:


model.fit(X_train, y_train ,batch_size = 10, epochs = 10, verbose=1)


# In[94]:


Predictions=model.predict(X_test)
Predictions=TargetVarScaler.inverse_transform(Predictions)
y_test_orig=TargetVarScaler.inverse_transform(y_test)
Test_Data=PredictorScaler.inverse_transform(X_test)
TestingData=pd.DataFrame(data=Test_Data, columns=Predictors)
TestingData['Price']=y_test_orig
TestingData['PredictedPrice']=Predictions
TestingData.head()


# In[95]:


model.history.history.values()


# In[ ]:




