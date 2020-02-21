# arima forecast for the power usage dataset
from math import sqrt
from numpy import split, mean, nan_to_num
from numpy import array, log
from pandas import read_csv
from matplotlib import pyplot


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


# one-step Holt Winter’s Exponential Smoothing forecast
def exp_smoothing_forecast(history, config):
	t,d,s,p,b,r = config
	# define model
	history = to_series(history)
	model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
	# fit model
	model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
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
		yhat_sequence = exp_smoothing_forecast(history, config)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	predictions = array(predictions)
	# evaluate predictions days for each week
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores


# grid search configs
def grid_search(data, cfg_list, parallel=False):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(evaluate_model)(data, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [evaluate_model(data, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of exponential smoothing configs to try
def exp_smoothing_configs(seasonal=[None]):
	models = list()
	# define config lists
	t_params = ['add', 'mul', None]
	d_params = [True, False]
	s_params = ['add', 'mul', None]
	p_params = seasonal
	b_params = [True, False]
	r_params = [True, False]
	# create config instances
	for t in t_params:
		for d in d_params:
			for s in s_params:
				for p in p_params:
					for b in b_params:
						for r in r_params:
							cfg = [t,d,s,p,b,r]
							models.append(cfg)
	return models




def exponential_smoothing(data, take_best=False):
	data = data.values
	# data split
	# model configs
	cfg_list = exp_smoothing_configs(seasonal=[52])
	# grid search
	
	if take_best == True:
		cfg_list = ['mul', False, 'add', 52, False, False]
		print(cfg_list)
		score, scores = evaluate_model(data, cfg_list)
		weeks = ["Wk" + str(i) for i in range(1,9)]
		results = score
		# summarize scores
		summarize_scores('exp_smoothing', score, scores)
		# plot scores
		pyplot.plot(weeks, scores, marker='o', label='exp_smoothing')
		# show plot
		pyplot.legend()
		pyplot.show()
		return results
    
    

	scores = grid_search(data, cfg_list)
	print('done')
	# list top 3 configs
	#print(scores[:3])
	for cfg, error in scores[:1]:
		if take_best==True: break
		print(cfg, error)
        
	return scores[0]
    
 
#'''
if __name__ == '__main__':
	# load dataset
	REPO_URL = 'https://raw.githubusercontent.com/nicholasrichers/Desafio-Cola-Cola-Sofazao/master/Datathon_Peta/datasets/'
	series = read_csv(REPO_URL + 'trainDF.csv', sep=',',
                       infer_datetime_format=True,
                       parse_dates=['Datetime'],
                       index_col=['Datetime'])
    #series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
	result = exponential_smoothing(series, take_best=True)
       
#'''

#cfg_list = [['mul', False, 'add', 52, False, False],
#['mul', False, 'add', 52, False, True],
#['add', False, 'add', 52, False, False]]

