#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 23:55:42 2020

@author: nicholasrichers
"""

##########
# File: arima_model.py
# Description:
#    Test Harness Arima Univ
##########


# arima forecast for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA




#from class_transform_dataset import Transform_Dataset

from numpy import mean
def mean_absolute_percentage_error(y_true, y_pred): 
    return mean(abs((y_true - y_pred) / y_true)) * 100


def split_dataset(data):
	# split into standard quarters
	train, test = data[8:-52], data[-52:]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/13))
	test = array(split(test, len(test)/13))
	return train, test




# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mape = mean_absolute_percentage_error(actual[:, i], predicted[:, i])
		# calculate rmse
		#rmse = sqrt(mse)
		# store
		scores.append(mape)
	# calculate overall MAPE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			(actual[row, col] - predicted[row, col])**2
			s +=  mean_absolute_percentage_error(actual[row, col], predicted[row, col])
	score =  (s / (actual.shape[0] * actual.shape[1]))
	return score, scores


# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))
#--------

# evaluate a single model
def evaluate_model(model_func, train, test):
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = model_func(history)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	predictions = array(predictions)
	# evaluate predictions days for each week
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores


#--------

# convert windows of quartely multivariate data into a series of total power
def to_series(data):
	# extract just the total power from each week
	series = [week[:, 0] for week in data]
	# flatten into a single series
	series = array(series).flatten()
	return series

# arima forecast
def arima_forecast(history):
	# convert history into a univariate series
	series = to_series(history)
	# define the model
	model = ARIMA(series, order=(4,0,0))
	# fit the model
	model_fit = model.fit(disp=False)
	# make forecast
	yhat = model_fit.predict(len(series), len(series)+12)
	return yhat


#--------



def arima(dataset):
    # split into train and test
    train, test = split_dataset(dataset.values)
    
    # define the names and functions for the models we wish to evaluate
    models = dict()
    models['arima'] = arima_forecast
    
    
    # evaluate each model
    weeks = ["Wk" + str(i) for i in range(1,14)]
    model_score = {}
    
    for name, func in models.items():
    	# evaluate and get scores
    	score, scores = evaluate_model(func, train, test)
    	# summarize scores
    	summarize_scores(name, score, scores)
    	model_score[name] = score
    	# plot scores
    	pyplot.plot(weeks, scores, marker='o', label=name)

    # show plot
    pyplot.legend()
    pyplot.show()
    
    std = 0
    return (min(model_score.values()), std)






#'''

if __name__ == '__main__':
    ###### Setup
    REPO_URL = 'https://raw.githubusercontent.com/nicholasrichers/Desafio-Cola-Cola-Sofazao/master/Datathon_Peta/datasets/'
    dataset = read_csv(REPO_URL + 'trainDF.csv', sep=',',
                       infer_datetime_format=True,
                       parse_dates=['Datetime'],
                       index_col=['Datetime'])
    

    
    model_scores = arima(dataset)
#'''




