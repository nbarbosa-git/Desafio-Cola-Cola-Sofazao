#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 02:58:28 2020

@author: nicholasrichers
"""

##########
# File: baseline_model.py
# Description:
#    Test Harness Modelos lineares recursivos
##########


# recursive multi-step forecast with linear algorithms
from math import sqrt
from numpy import split
from numpy import array
from numpy import log, std
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
from sklearn.svm import SVR


from sklearn.pipeline import FeatureUnion
from skits.pipeline import ForecasterPipeline
from skits.feature_extraction import AutoregressiveTransformer
from skits.feature_extraction import SeasonalTransformer
from skits.preprocessing import ReversibleImputer
from skits.preprocessing import DifferenceTransformer
from numpy import newaxis

import warnings
warnings.filterwarnings("ignore")

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
			s+= (( log(actual[row, col]) - log(predicted[row, col]))**2)
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores


# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))
#-------


def get_models(models=dict()):
	# linear models
	models['lr'] = LinearRegression()
	models['lasso'] = Lasso()
	models['ridge'] = Ridge()
	models['en'] = ElasticNet()
	models['huber'] = HuberRegressor()
	models['svmr'] = SVR()
	#models['lars'] = Lars()
	#models['llars'] = LassoLars()
	#models['sgd'] = SGDRegressor(max_iter=1000000, tol=1e-3)
	#models['pa'] = PassiveAggressiveRegressor(max_iter=1000000, tol=1e-3)
	#models['ranscac'] = RANSACRegressor()
	print('Defined %d models' % len(models))
	return models
    
	print('Defined %d models' % len(models))
	return models



# create a feature preparation pipeline for a model
def make_pipeline_(model):
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


# create a feature preparation pipeline for a model
def make_pipeline(model):
    steps = [('features', FeatureUnion([
        ('seasonal_features', SeasonalTransformer(seasonal_period=1)),
        ('ar_features', AutoregressiveTransformer(num_lags=1))
    ]))]

    #steps = list() 
    steps.append(('post_feature_imputer', ReversibleImputer()))
    # standardization    
    steps.append(('standardize', StandardScaler()))
    # normalization
    steps.append(('normalize', MinMaxScaler()))
    # the model
    steps.append(('model', model))
    # create pipeline
    #pipeline = ForecasterPipeline(steps=steps)
    pipeline = ForecasterPipeline([
    ('pre_differencer', DifferenceTransformer(period=1)),
    ('pre_diff_imputer', ReversibleImputer()),
    ('pre_day_differencer', DifferenceTransformer(period=1)),
    ('pre_day_diff_imputer', ReversibleImputer()),
    ('pre_scaler', StandardScaler()),
    ('features', FeatureUnion([
        ('ar_features', AutoregressiveTransformer(num_lags=1)),
        ('seasonal_features', SeasonalTransformer(seasonal_period=1)),
    ])),
    ('post_feature_imputer', ReversibleImputer()),
    ('post_feature_scaler', StandardScaler()),
    ('model', LinearRegression(fit_intercept=False))])

    
    return pipeline

# make a recursive multi-step forecast
def forecast(model, input_x, n_input):
	yhat_sequence = list()
	input_data = [x for x in input_x]
	for j in range(8):
		# prepare the input data
		X = array(input_data[-n_input:]).reshape(1, n_input)
		# make a one-step forecast
		yhat = model.predict(X)[0]
		# add to the result
		yhat_sequence.append(yhat)
		# add the prediction to the input
		input_data.append(yhat)
	return yhat_sequence

# convert windows of weekly multivariate data into a series of total power
def to_series(data):
	# extract just the total power from each week
	series = [week[:,0] for week in data] #week[:, 0]
	features = [week[:,1:] for week in data] #week[:, 0]
	# flatten into a single series
	series = array(series).flatten()
	features = array(features).reshape(series.shape[0], 17)
	return series, features




# convert history into inputs and outputs
def to_supervised(history, n_input, output_ix):
	# convert history to a univariate series
	data, features = to_series(history)
	X, y = list(), list()
	ix_start = 0
	# step over the entire history one time step at a time
	for i in range(len(data)):
		# define the end of the input sequence
		ix_end = ix_start + n_input
		ix_output = ix_end + output_ix
		# ensure we have enough data for this instance
		if ix_output < len(data):
			#lags = data[ix_start:ix_end]
			#feat = features[ix_end-1, :]
			#X.append(concatenate((lags, feat), axis=0))
			X.append(data[ix_start:ix_end])
			y.append(data[ix_output])
		# move along one time step
		ix_start += 1
	return array(X), array(y)



# fit a model and make a forecast
def sklearn_predict(model, history, n_input):
	yhat_sequence = list()
	# fit a model for each forecast day
	for i in range(8):
		# prepare data
		train_x, train_y = to_supervised(history, n_input, i)
		# make pipeline
		pipeline = make_pipeline(model)
		# fit the model
		c = train_x[:,0].copy()[:, newaxis]
		cy = train_y.copy()[:, newaxis]
		pipeline.fit(cy, train_y)
		# forecast
		x_input = array(train_x[-1, :]).reshape(1,train_x.shape[1])
		yhat = pipeline.predict(x_input, to_scale=True, refit=True)
		#yhat = pipeline.predict(x_input)[0]
		# store
		yhat_sequence.append(yhat)
	return yhat_sequence



# evaluate a single model
def evaluate_model(model, train, test, n_input):
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = sklearn_predict(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	predictions = array(predictions)
	# evaluate predictions days for each week
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores


#--------
def linear_recursive(dataset):
    # split into train and test
    train, test = split_dataset(dataset.values)
    # prepare the models to evaluate
    models = get_models()
    n_input = 1
    # evaluate each model
    weeks = ["Wk" + str(i) for i in range(1,9)]
    results = dict()
    for name, model in models.items():
        # evaluate and get scores
        score, scores = evaluate_model(model, train, test, n_input)
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
    
    
    model_scores = linear_recursive(dataset)
#'''




#'''
