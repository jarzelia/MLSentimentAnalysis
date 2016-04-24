# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:00:17 2016

@author: Pat
"""
import numpy as np


def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def computeCost(theta, X, y, l):
    
    # m - Number of training examples
    m = X.shape[0]
    
    # p - Regularization parameter, p_theta - theta indexed from 1
    p_theta = theta[1:np.size(theta):1]
    p_theta = np.insert(p_theta, 0, 0, axis=0)
    p = (l/(2*m)) * (np.dot(np.transpose(p_theta), p_theta))
    
    # h - Hypothesis function
    h = sigmoid(np.dot(X, theta))
    
    # yt - y'
    yt = -np.transpose(y)
    
    # lh - log(h)
    lh = np.log(h)
    
    # negy - (1 -y)'
    negy = np.transpose(1 - np.array(y))
    
    # negh - log(1 - h)
    negh = np.log(1 - np.array(h))
    
    # Cost function
    J = (1/m) * (np.dot(yt, lh) - np.dot(negy, negh)) + p
    
    return J
    
def computeGrad(theta, X, y, l):
    grad = np.zeros(np.shape(theta));
    
    # m - Number of training examples
    m = X.shape[0]
    
    # p - Regularization parameter
    p_theta = theta[1:np.size(theta):1]
    p_theta = np.insert(p_theta, 0, 0, axis=0)
    p = (l/m) * p_theta
    
    # h - Hypothesis function
    h = sigmoid(np.dot(X, theta))
        
    # Gradient
    grad = (1/m) * np.dot(np.transpose(X), (h-y)) + p
    
            
    return grad