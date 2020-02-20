#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 00:22:12 2020

@author: nicholasrichers
"""

from fbprophet import Prophet


from math import sqrt
from numpy import split
from numpy import array
from numpy import log, std
from numpy import concatenate
import pandas as pd
import numpy as np




# calculo do erro do modelo

def calculate_baseline_errors(df, prediction_size):
    """Calculate MAPE of the forecast.
    
       Args:
           df: joined dataset with 'y' and 'yhat' columns.
           prediction_size: number of days at the end to predict.
    """
    
    # Make a copy
    df = df.copy()
    
    # Now we calculate the values of e_i and p_i according to the formulas given in the article above.
    df['e'] = df['y'] - df['yhat']
    df['p'] = 100 * df['e'] / df['y']
    
    # Recall that we held out the values of the last `prediction_size` days
    # in order to predict them and measure the quality of the model. 
    
    # Now cut out the part of the data which we made our prediction for.
    predicted_part = df[-prediction_size:]
    
    # Define the function that averages absolute error values over the predicted part.
    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))
       
    # Now we can calculate MAPE and MAE and return the resulting dictionary of errors.
    return {'Baseline model Prophet MAPE': error_mean('p')}


def make_comparison_dataframe(historical, forecast):
    """Join the history with the forecast.
    
       The resulting dataset will contain columns 'yhat', 'yhat_lower', 'yhat_upper' and 'y'.
    """
    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))








def baseline_prophet(dataset):
    prediction_size=40
    
    X_prophet = pd.DataFrame({ 'ds': dataset.index.values, 'y':  dataset.iloc[:, 0]})
    
    
    
    train_df = X_prophet[:-prediction_size]
    
    
    m = Prophet(interval_width=0.95)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=40,freq='W')
    forecast = m.predict(future)
    #forecast.tail()


    # Print do forecast
    return m.plot(forecast, uncertainty=True, plot_cap=True)


