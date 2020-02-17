
























			s+= (( log(actual[row, col]) - log(predicted[row, col]))**2)
			X.append(data[ix_start:ix_end])
			y.append(data[ix_end])
		# add the prediction to the input
		# add to the result
		# calculate mse
		# calculate rmse
		# define the end of the input sequence
		# ensure we have enough data for this instance
		# get real observation and add to history for predicting the next week
		# make a one-step forecast
		# move along one time step
		# predict the week
		# prepare the input data
		# store
		# store the predictions
		for col in range(actual.shape[1]):
		history.append(test[i, :])
		if ix_end < len(data):
		input_data.append(yhat)
		ix_end = ix_start + n_input
		ix_start += 1
		mse = mean_squared_log_error(actual[:, i], predicted[:, i])
		predictions.append(yhat_sequence)
		rmse = sqrt(mse)
		scores.append(rmse)
		X = array(input_data[-n_input:]).reshape(1, n_input)
		yhat = model.predict(X)[0]
		yhat_sequence = sklearn_predict(model, history, n_input)
		yhat_sequence.append(yhat)
	# calculate an RMSE score for each day
	# calculate overall RMSE
	# convert history to a univariate series
	# create pipeline
	# evaluate predictions days for each week
	# extract just the total power from each week
	# fit the model
	# flatten into a single series
	# history is a list of weekly data
	# linear models
	# make pipeline
	# normalization
	# predict the week, recursively
	# prepare data
	# restructure into windows of weekly data
	# split into standard periods
	# standardization
	# step over the entire history one time step at a time
	# the model
	# walk-forward validation over each week
	#models['pa'] = PassiveAggressiveRegressor(max_iter=1000000, tol=1e-3)
	#models['ranscac'] = RANSACRegressor()
	data = to_series(history)
	for i in range(actual.shape[1]):
	for i in range(len(data)):
	for i in range(len(test)):
	for j in range(8):
	for row in range(actual.shape[0]):
	history = [x for x in train]
	input_data = [x for x in input_x]
	ix_start = 0
	models['en'] = ElasticNet()
	models['huber'] = HuberRegressor()
	models['lars'] = Lars()
	models['lasso'] = Lasso()
	models['llars'] = LassoLars()
	models['lr'] = LinearRegression()
	models['ridge'] = Ridge()
	models['sgd'] = SGDRegressor(max_iter=1000000, tol=1e-3)
	models['svmr'] = SVR()
	pipeline = make_pipeline(model)
	pipeline = Pipeline(steps=steps)
	pipeline.fit(train_x, train_y)
	predictions = array(predictions)
	predictions = list()
	print('%s: [%.3f] %s' % (name, score, s_scores))
	print('Defined %d models' % len(models))
	return array(X), array(y)
	return models
	return pipeline
	return score, scores
	return score, scores
	return series
	return train, test
	return yhat_sequence
	return yhat_sequence
	s = 0
	s_scores = ', '.join(['%.1f' % s for s in scores])
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	scores = list()
	series = [week[:, 0] for week in data]
	series = array(series).flatten()
	steps = list()
	steps.append(('model', model))
	steps.append(('normalize', MinMaxScaler()))
	steps.append(('standardize', StandardScaler()))
	test = array(split(test, len(test)/8))
	train = array(split(train, len(train)/8))
	train, test = data[2:-40], data[-40:]
	train_x, train_y = to_supervised(history, n_input)
	X, y = list(), list()
	yhat_sequence = forecast(pipeline, train_x[-1, :], n_input)
	yhat_sequence = list()
    
    
    
    
        
                       index_col=['Datetime'])
                       infer_datetime_format=True,
                       parse_dates=['Datetime'],
        # evaluate and get scores
        # plot scores
        # summarize scores
        pyplot.plot(weeks, scores, marker='o', label=name)
        results[name] = score
        score, scores = evaluate_model(model, train, test, n_input)
        summarize_scores(name, score, scores)
    # evaluate each model
    # prepare the models to evaluate
    # show plot
    # split into train and test
    ###### Setup
    #Xt = Transform_Dataset(dataset)
    #Xt.compose(get_df(),compose_values.resid)
    #Xt.decompose()
    #Xt.df.head(4)
    dataset = read_csv(REPO_URL + 'trainDF.csv', sep=',',
    for name, model in models.items():
    model_scores = linear_recursive(dataset)
    models = get_models()
    n_input = 4
    pyplot.legend()
    pyplot.show()
    REPO_URL = 'https://raw.githubusercontent.com/nicholasrichers/Desafio-Cola-Cola-Sofazao/master/Datathon_Peta/datasets/'
    results = dict()
    return mean(abs((y_true - y_pred) / y_true)) * 100
    return results
    train, test = split_dataset(dataset.values)
    weeks = ["Wk" + str(i) for i in range(1,9)]
"""
"""
#    Test Harness Modelos lineares recursivos
# -*- coding: utf-8 -*-
# convert history into inputs and outputs
# convert windows of weekly multivariate data into a series of total power
# create a feature preparation pipeline for a model
# Description:
# evaluate a single model
# evaluate one or more weekly forecasts against expected values
# File: baseline_model.py
# fit a model and make a forecast
# make a recursive multi-step forecast
# prepare a list of ml models
# recursive multi-step forecast with linear algorithms
# summarize scores
#!/usr/bin/env python3
##########
##########
#'''
#'''
#'''
#-------
#--------
@author: nicholasrichers
Created on Sat Feb 15 02:58:28 2020
def evaluate_forecasts(actual, predicted):
def evaluate_model(model, train, test, n_input):
def forecast(model, input_x, n_input):
def get_models(models=dict()):
def linear_recursive(dataset):
def make_pipeline(model):
def mean_absolute_percentage_error(y_true, y_pred): 
def sklearn_predict(model, history, n_input):
def split_dataset(data):
def summarize_scores(name, score, scores):
def to_series(data):
def to_supervised(history, n_input):
from math import sqrt
from matplotlib import pyplot
from numpy import array
from numpy import log, std
from numpy import mean
from numpy import split
from pandas import read_csv
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
if __name__ == '__main__':