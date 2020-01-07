# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("input/hiring.csv")

dataset["experience"].fillna(0, inplace=True)

dataset["test_score"].fillna(dataset["test_score"].mean(), inplace=True)


#X = dataset.iloc[:,:-1]

def mapTextExp_to_Numeric(word):
    word_dict = {'one':1,  'two':2, 'three':3, 'four':4, 'five':5, 'six':6,
                 'seven':7, 'eight':8, 'nine':9,
                 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

dataset['experience'] = dataset['experience'].apply(lambda x: mapTextExp_to_Numeric(x))

dataset.to_csv("input/trained_hiring.csv", index=False)




