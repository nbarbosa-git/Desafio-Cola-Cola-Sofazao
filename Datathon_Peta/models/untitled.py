




def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[0]):
    	for j in range(actual.shape[1]):
    		# calculate mse
    		mse = mean_squared_log_error(actual[i, j], predicted[i, j])
    		# calculate rmse
    		rmse = sqrt(mse)
    		# store
    		scores.append(rmse)
	# calculate overall RMSE