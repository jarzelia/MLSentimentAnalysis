# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:11:22 2016

@author: Pat
"""
import numpy as np
from lrFunctions import sigmoid

def predict(theta, X):
    m, n = X.shape
    
    p = np.zeros(shape = (m, 1))
    
    for i in range (1, m):
        if sigmoid(np.dot(X[i], theta)) > 0.5:
            p[i, 0] = 1
            
    