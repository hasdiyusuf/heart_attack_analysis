# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:33:32 2022

@author: DELL
"""

import os 
import pandas as pd 
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#%% step 1 data loading
PATH = os.path.join(os.getcwd(),'dataset','heart.csv')
MODEL_PATH = os.path.join(os.getcwd(),'saved_model','model.pkl')
MMS_SCALER_PATH = os.path.join(os.getcwd(),'saved_model', 'mms_scaler.pkl')
OHE_SCALER_PATH = os.path.join(os.getcwd(),'saved_model', 'ohe_scaler.pkl')


df = pd.read_csv(PATH) # create DataFrame

#%% step 2 data inspection

# get dataframe information
df.head()
df.shape
df.columns 
df.info()
df.describe().T

# visualize boxplot to check any outliers
df.boxplot()

# checking null 
df.isna().sum()

#%% step 3 data cleaning
#%% step 4 feature scalling 

# split the features and the target label 
x = df.drop('output', axis = 1)
y = df['output']


# feature scalling - Min Max Scaler
mms = MinMaxScaler()
x_scaled = mms.fit_transform(x)

pickle.dump(mms, open(MMS_SCALER_PATH, 'wb'))  # save scaler

# one hot encoding
ohe = OneHotEncoder(sparse = False)
y_scaled = ohe.fit(np.expand_dims(y, axis =-1))

pickle.dump(y_scaled, open(OHE_SCALER_PATH,'wb')) # save one hot encoder scaler

# split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y,
                                                    test_size = 0.3,
                                                    random_state = 123)




#%% model prediction/evaluation

lr = LogisticRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
y_true = y_test

# Logistic Regression is used for this model prediction because it is a classification problem

print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))

# from the report, the model avhieve to predict 80% accuracy of the test set
 

#%% save nodel
pickle.dump(lr, open(MODEL_PATH,'wb'))
