#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:37:00 2020

@author: nicholasrichers
"""

##########
# File: baseline_model.py
# Description:
#    Test Harness Modelo Baseline 1
##########




 
# naive forecast strategies for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array
from numpy import log
from pandas import read_csv
from sklearn.metrics import mean_squared_log_error
from matplotlib import pyplot


#from class_transform_dataset import Transform_Dataset

from numpy import mean
def mean_absolute_percentage_error(y_true, y_pred): 
    return mean(abs((y_true - y_pred) / y_true)) * 100


def split_dataset(data):
	# split into standard periods
	train, test = data[2:-40], data[-40:]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/8))
	test = array(split(test, len(test)/8))
	return train, test




# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_log_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (( log(actual[row, col]) - log(predicted[row, col]))**2)
	score = sqrt((s / (actual.shape[0] * actual.shape[1])))
	return score, scores




# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.2f' % s for s in scores])
	print('%s: [%.2f] %s' % (name, score, s_scores))


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


# daily persistence model
def daily_persistence(history):
	# get the data for the prior week
	last_week = history[-1]
	# get the total active power for the last day
	value = last_week[-1, 0]
	# prepare 7 day forecast
	forecast = [value for _ in range(8)]
	return forecast


# weekly persistence model
def weekly_persistence(history):
	# get the data for the prior week
	last_week = history[-1]
	return last_week[:, 0]


# week one year ago persistence model
def week_one_year_ago_persistence(history):
	# get the data for the prior week
	last_week = history[-4]
	return last_week[:, 0]




def baseline(dataset):
    # split into train and test
    train, test = split_dataset(dataset.values)
    
    # define the names and functions for the models we wish to evaluate
    models = dict()
    models['weekly'] = daily_persistence
    models['quarterly'] = weekly_persistence
    models['yearly'] = week_one_year_ago_persistence
    
    
    # evaluate each model
    weeks = ["Wk" + str(i) for i in range(1,9)]
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


'''

if __name__ == '__main__':
    ###### Setup
    REPO_URL = 'https://raw.githubusercontent.com/nicholasrichers/Desafio-Cola-Cola-Sofazao/master/Datathon_Peta/datasets/'
    dataset = read_csv(REPO_URL + 'trainDF.csv', sep=',',
                       infer_datetime_format=True,
                       parse_dates=['Datetime'],
                       index_col=['Datetime'])
    
    #Xt = Transform_Dataset(dataset)
    #Xt.decompose()
    #Xt.compose(get_df(),compose_values.resid)
    #Xt.df.head(4)
    
    
    model_scores = baseline(dataset)
'''