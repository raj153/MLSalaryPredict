# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:59:53 2020

@author: rgottimu
"""
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import pickle

#trained_data=os.environ.get("trained_hiring.csv")

dataset = pd.read_csv("input/trained_hiring.csv")

X = dataset.iloc[:, :-1]
y = dataset.iloc[:,3]
regressor = LinearRegression()


regressor.fit(X, y)

#Saving model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))

#Loading model to compare results
model = pickle.load(open("model.pkl","rb"))
print(model.predict([[11, 7 ,8]]))