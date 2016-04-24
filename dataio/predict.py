# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:11:22 2016

@author: Pat
"""
import numpy as np
from lrFunctions import sigmoid

def predict(theta, X):
    m, n = X.shape
    
    p = np.zeros(shape=(m, 1))
    h = sigmoid(np.dot(X, theta))
    
    for i in range (0, h.shape[0]):
        if h[i] >= 0.5:
            p[i, 0] = 1;
        
    return p
            
    