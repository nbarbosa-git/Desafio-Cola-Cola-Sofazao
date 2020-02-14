#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 21:18:52 2020

@author: nicholasrichers
"""



# naive forecast strategies for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot



###### Setup
REPO_URL = 'https://raw.githubusercontent.com/nicholasrichers/Desafio-Cola-Cola-Sofazao/master/Datathon_Peta/datasets/'
X = read_csv(REPO_URL + 'processedDF.csv', sep=',',
                   infer_datetime_format=True,
                   parse_dates=['Datetime'],
                   index_col=['Datetime'])

#reorder columns
X = X[['Total', 'SM', 'ROUTE', 'INDIRETOS',  'OUTROS',
 'Day_gregoriano', 'Week_Month', 'Week_Year',
 'Month_gregoriano', 'Year', 'holidays', 'Next_holiday',
 'temperatura', 'Ajuste_ipca', 'PMC',
  'Massa.Renda', 'Renda', 'Desemprego']]



X.to_csv(r'processedDF.csv', header=True)


X_train = X.iloc[:-8, :]
X_test = X.iloc[-8:, :]


X_train.to_csv(r'trainDF.csv', header=True)
X_test.to_csv(r'testDF.csv', header=True)


