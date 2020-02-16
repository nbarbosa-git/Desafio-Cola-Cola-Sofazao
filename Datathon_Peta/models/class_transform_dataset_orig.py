#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:37:00 2020

@author: nicholasrichers
"""

##########
# File: class_transform_dataset.py
# Description:
#    Classe para manipular o dataset
##########

#import pandas
from statsmodels.tsa.seasonal import STL

class Transform_Dataset:
    
    def __init__(self, df, Multi=False):
        self.df = df
        if Multi==False: self.cols =  ['Total']
        else: self.cols = ['Total', 'SM', 'ROUTE', 'INDIRETOS',  'OUTROS']
    
    def decompose(self):
        for col in self.cols:
          decompose_values= STL(self.df[col], period=52).fit()
          self.df[col] = decompose_values.resid
    
    def compose(self, orig, predict):
        for col in self.cols:
          compose_values = STL(orig[col], period=52).fit()
          self.df[col] = compose_values.trend + compose_values.seasonal + predict[col]