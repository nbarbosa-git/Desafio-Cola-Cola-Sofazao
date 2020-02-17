#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 03:32:15 2020

@author: nicholasrichers
"""

##########
# File: baseline_model.py
# Description:
#    Test Harness Modelos ensemble recursivos
##########


# direct multi-step forecast by day
from math import sqrt
from numpy import split
from numpy import array
#from numpy import append
from numpy import log
from numpy import concatenate
from pandas import read_csv
from sklearn.metrics import mean_squared_log_error
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor

from sklearn.base import clone
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor


from numpy import mean
def mean_absolute_percentage_error(y_true, y_pred): 
    return mean(abs((y_true - y_pred) / y_true)) * 100


def split_dataset(data):
	# split into standard periods
	train, test = data.iloc[:-40, :], data.iloc[-40:, :]
	# restructure into windows of weekly data
	#train = array(split(train, len(train)/8))
	#test = array(split(test, len(test)/8))
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
			s+= (( log(actual[row, col]) - log(predicted[row, col]))**2)
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores




# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))
#-------
# prepare a list of ml models
def get_models(models=dict()):
	# non-linear models
	#models['knn'] = KNeighborsRegressor(n_neighbors=8)
	#models['cart'] = DecisionTreeRegressor()
	#models['extra'] = ExtraTreeRegressor()
	models['svmr'] = SVR()
	# # ensemble models
	#n_trees = 100 #500
	#models['ada'] = AdaBoostRegressor(n_estimators=n_trees)
	#models['bag'] = BaggingRegressor(n_estimators=n_trees)
	#models['rf'] = RandomForestRegressor(n_estimators=n_trees)
	#models['et'] = ExtraTreesRegressor(n_estimators=n_trees)
	#models['gbm'] = GradientBoostingRegressor(n_estimators=n_trees)
    
    #inserir XGBOOST e LGBM (configs Mario)
    
	print('Defined %d models' % len(models))
	return models

# create a feature preparation pipeline for a model
def make_pipeline(model):
	steps = list()
	# standardization
	steps.append(('standardize', StandardScaler()))
	# normalization
	steps.append(('normalize', MinMaxScaler()))
	# the model
	steps.append(('model', model))
	# create pipeline
	pipeline = Pipeline(steps=steps)
	return pipeline

#----

# convert history into inputs and outputs
def to_supervised(history, output_ix):
	X, y = list(), list()
	# step over the entire history one time step at a time
	for i in range(len(history)-1):
		lags = history[i][:,0]
		# to get the other features from last timestep
		features = history[i][7 , 1:]
		features.reshape(1,-1)[0]
		X.append(concatenate((lags, features), axis=0))
		y.append(history[i + 1][output_ix,0])
	print(len(X))
	return array(X), array(y)



# fit a model and make a forecast
def sklearn_predict(model, history):
	yhat_sequence = list()
	# fit a model for each forecast day
	for i in range(8): ##
		# prepare data
		train_x, train_y = to_supervised(history, i)
		# make pipeline
		pipeline = make_pipeline(model)
		# fit the model
		pipeline.fit(train_x, train_y)
		# forecast
		x_input = array(train_x[-1, :]).reshape(1,train_x.shape[1]) ##
		yhat = pipeline.predict(x_input)[0]
		# store
		yhat_sequence.append(yhat)
	return yhat_sequence

# evaluate a single model
def evaluate_model(model, train, test):
	# history is a list of weekly data
	history = train #[x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = sklearn_predict(model, history)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next timestep
		history.append(test[i, :])
	predictions = array(predictions)
	# evaluate predictions days for each week
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores

#--------
def linear_direct(dataset):
    # split into train and test
    train, test = split_dataset(dataset)
    # prepare the models to evaluate
    models = get_models()
    # evaluate each model
    weeks = ["Wk" + str(i) for i in range(1,9)]
    results = dict()
    for name, model in models.items():
        # evaluate and get scores
        score, scores = evaluate_model(model, train, test)
        results[name] = score
        # summarize scores
        summarize_scores(name, score, scores)
        # plot scores
        pyplot.plot(weeks, scores, marker='o', label=name)
    # show plot
    pyplot.legend()
    pyplot.show()
        
    return results
    
#'''

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
    
    
    model_scores = linear_direct(dataset)
#'''
