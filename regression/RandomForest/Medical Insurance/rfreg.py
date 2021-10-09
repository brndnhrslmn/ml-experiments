"""
Created on Wed Sep 29 11:59:16 2021

@author: Brandon.Herselman
"""

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

cwd = os.getcwd()
dataPath = cwd + '\\regression\\RandomForest\\Medical Insurance\\data\\insurance.csv'
modelPath = os.path.join(cwd,'regression\\RandomForest\\Medical Insurance\\model', 'rf_insurance.pkl')
df = pd.read_csv(dataPath)

num_list = []
cat_list = []

for column in df:
    if is_numeric_dtype(df[column]):
        num_list.append(column)
    elif is_string_dtype(df[column]):
        cat_list.append(column)

df['log_expenses'] = np.log2(df['expenses'] + 1)

for i in cat_list:
    df[i] = LabelEncoder().fit_transform(df[i])
df.head()

X = df.iloc[:, 0:6]
y = df.log_expenses
x_train, x_val, y_train, y_val = train_test_split(X,y, shuffle=True, test_size=0.2, random_state=13903)

model = RandomForestRegressor(random_state=1)
model.fit(x_train, y_train)
y_hat = model.predict(x_val)

print('MAE: {:,.3f}'.format(mean_absolute_error(y_val, y_hat)))
print('MSE: {:,.3f}'.format(mean_squared_error(y_val, y_hat)))
print('RMSE: {:,.3f}'.format(np.sqrt(mean_squared_error(y_val, y_hat))))
print('R Squared: {:,.2f}'.format(model.score(x_val, y_val)))

cwd = os.getcwd()
modelPath = os.path.join(cwd,'regression\\RandomForest\\Medical Insurance\\model', 'rf_insurance.pkl')
with open(modelPath, 'wb') as file:
	pickle.dump(model, file)