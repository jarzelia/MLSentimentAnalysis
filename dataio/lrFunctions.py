# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:00:17 2016

@author: Pat
"""
import numpy as np


def sigmoid(X):
    z = 1.0 / ( 1.0 + np.exp(-1.0 * X) )
    
    return z
    
def computeCost(theta, X, y):
    
    # Get number of training examples
    m = X.shape[0]
    
    h = sigmoid(np.dot(X, theta))
    
    yt = -np.transpose(y)
    lh = np.log(h)
    negy = np.transpose(1 - np.array(y))
    negh = np.log(1 - np.array(h))
    
    J = (1./m) * (np.dot(yt, lh) - np.dot(negy, negh))
    
    return J
    
def computeGrad(theta, X, y):
    
    # Get number of training examples
    m = X.shape[0]
    theta = np.reshape(theta, (len(theta), 1))
    h = sigmoid(np.dot(X, theta))
    
    grad = (1./m) * (np.dot(np.transpose(sigmoid(h) - y), X))
    return  grad
    