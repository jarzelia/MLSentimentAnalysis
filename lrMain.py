# -*- coding: utf-8 -*-

from lrFunctions import computeCost, computeGrad, predict
from cleanReview import cleanReview
from scipy import optimize

import pandas as pd
import numpy as np
import math, os

validationFlag = False
posReviews = []        # Holds train/pos/ reviews
negReviews = []        # Holds train/neg/ reviews
testReviews = []       # Holds test/ reviews

trainPosPath = "train/pos/"
trainNegPath = "train/neg/"
testPath = "test/"

m_train = len( os.listdir(trainPosPath) ) + len( os.listdir(trainNegPath) )

print ("\nCleaning reviews...")

print ("Cleaning positive training set...", end='\t')
posReviews = cleanReview(trainPosPath)
print ("Done!")

print ("Cleaning negative training set...", end='\t')
negReviews = cleanReview(trainNegPath)
print ("Done!")

print ("Cleaning test set...", end='\t\t\t')
testReviews = cleanReview(testPath)
print ("Done!\n")

# Append posReviews and negReviews together
allReviews = posReviews + negReviews

# Create indices for test set
test_id = list(range(len( os.listdir(testPath) )))

# Create labels (class) for training set
train_labels = []
for i in range(len(allReviews)):
    if i > math.floor(len(allReviews)/2):
        train_labels.append(0)
    else:
        train_labels.append(1)


print ("Creating bag of words...", end='\t\t')

from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, ngram_range=(1, 1), preprocessor = None, stop_words = None, max_features = 5000)

X_train = vectorizer.fit_transform(allReviews)
X_train = X_train.toarray()

# Split the training set if validationFlag == true
if validationFlag == True:
    from sklearn.cross_validation import train_test_split
    X_train, X_validation, train_labels, validation_labels = train_test_split(X_train, train_labels, test_size=0.2, random_state=0)


print ("Done!\n")

# ================= Logistic Regression Test (ours) =================
print ("Training Logistic Regression...", end='\t\t')

# Set n - number of features in training set
n = X_train.shape[1]

# Initializing fitting parameters
init_theta = np.zeros((n + 1, 1))

# Append intercept term to our training features
X_train_1 = np.append( np.ones((X_train.shape[0], 1)), X_train, axis=1)


# Check initial Cost & Gradient
# Computing Cost
#cost = computeCost(init_theta, X_train_1, train_labels, 1)
#print ('Cost at initial theta (zeros): ' + str(cost))

# Computing Gradient
#grad = computeGrad(init_theta, X_train_1, train_labels)
#print ('Gradient at initial theta (zeros):' + str(grad))

# Optimizing fitting parameters
lmbda = 1
opt_theta = optimize.minimize(computeCost, init_theta, args=(X_1, train_labels, lmbda), method='BFGS', jac=computeGrad )
print ("Done!")

# Predicting the training set
p = predict(opt_theta[0], X_train_1)

accuracy = 0
for i in range(len(train_labels)):
    if p[i] == train_labels[i]:
        accuracy = accuracy + 1
accuracy = (accuracy / len(train_labels)) * 100
print ("Training Set Accuracy:\t\t\t" + str(accuracy) + "%")

# Predicting the validation set (if validationFlag == true)
if validationFlag == True:
    X_1 = np.append( np.ones((X_validation.shape[0], 1)), X_validation, axis=1)
    p = predict(opt_theta.x, X_1)
    
    accuracy = 0
    for i in range(len(validation_labels)):
        if p[i] == validation_labels[i]:
            accuracy = accuracy + 1
    accuracy = (accuracy / len(validation_labels)) * 100
    print ("Validation Set Accuracy:\t\t" + str(accuracy) + "%")

# Predict labels on test set
X_test = vectorizer.transform(testReviews)
X_test = X_test.toarray()

X_1_test = np.append( np.ones((X_test.shape[0], 1)), X_test, axis=1)
result = predict(opt_theta[0], X_1_test)

result = result.tolist()

# Convert to something we can write to .csv
for i, val in enumerate(result):
    result[i] = math.floor(val[0])
    

# Writing to sample2.csv
output = pd.DataFrame( data = {"id": test_id, "labels":result})
output.to_csv("sample.csv", index=False)
