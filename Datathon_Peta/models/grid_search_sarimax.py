#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 05:43:12 2020

@author: nicholasrichers
"""

# arima forecast for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array, log
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import  mean_squared_log_error
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

import warnings
warnings.filterwarnings("ignore")


# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard periods
	train, test = data[2:-40], data[-40:]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/8))
	test = array(split(test, len(test)/8))
	return train, test


from numpy import mean
def mean_absolute_percentage_error(y_true, y_pred): 
    return mean(abs((y_true - y_pred) / y_true)) * 100

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
	# calculate overall RMSE
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



# convert windows of weekly multivariate data into a series of total power
def to_series(data):
	# extract just the total power from each week
	series = [week[:,0] for week in data]
	# flatten into a single series
	series = array(series).flatten()
	return series



# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	history = to_series(history)
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history)+7)
	return yhat




# evaluate a single model
def evaluate_model(dataset, config):
	# split into train and test
	train, test = split_dataset(dataset)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = sarima_forecast(history, config)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	predictions = array(predictions)
	# evaluate predictions days for each week
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores





# grid search configs
def grid_search(data, cfg_list, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal #[0,4,52]
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models


def sarimax(data, take_best=False):
	data = data.values
    # model configs
	cfg_list = sarima_configs(seasonal=[52])
    
	if take_best == True:
		cfg_list = [(2, 0, 0), (0, 1, 2, 52), 'c']
		#cfg_list = [(0, 0, 0), (0, 0, 0, 0), 'c']
		score, scores = evaluate_model(data, cfg_list)
		weeks = ["Wk" + str(i) for i in range(1,9)]
		results = score
		# summarize scores
		summarize_scores('SARIMAX', score, scores)
		# plot scores
		pyplot.plot(weeks, scores, marker='o', label='SARIMAX')
		# show plot
		pyplot.legend()
		pyplot.show()
		return results
    
	# grid search
	scores = grid_search(data, cfg_list)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		if take_best==True: break
		print(cfg, error)
	return scores[0]
        




if __name__ == '__main__':
	# load dataset
  REPO_URL = 'https://raw.githubusercontent.com/nicholasrichers/Desafio-Cola-Cola-Sofazao/master/Datathon_Peta/datasets/'
  series = read_csv(REPO_URL + 'trainDF.csv', sep=',',
                       infer_datetime_format=True,
                       parse_dates=['Datetime'],
                       index_col=['Datetime'])


  sarimax(series, take_best=True)

#Model[[(2, 0, 0), (0, 1, 2, 52), 'c']] 0.168
#Model[[(2, 0, 0), (1, 1, 2, 52), 'c']] 0.179
#Model[[(2, 0, 0), (0, 1, 2, 52), 't']] 0.181

