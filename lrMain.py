# -*- coding: utf-8 -*-

from lrFunctions import computeCost, computeGrad, predict
from cleanReview import cleanReview
from scipy import optimize

import pandas as pd
import numpy as np
import math, os

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
for i in range(m_train):
    if i > 12500:
        train_labels.append(0)
    else:
        train_labels.append(1)


print ("Creating bag of words...", end='\t\t')
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, ngram_range=(1, 2), preprocessor = None, stop_words = None, max_features = 5000)

train_data_features = vectorizer.fit_transform(allReviews)
train_data_features = train_data_features.toarray()

print ("Done!\n")

# ================= Random Forest Test (sklearn) =================
'''
print ("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit forest to training set
forest = forest.fit( train_data_features, train_labels)

test_data_features = vectorizer.transform(testReviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

# Writing to sample.csv
output = pd.DataFrame( data = {"id": test_id, "labels":result})
output.to_csv("sample.csv", index=False)
'''

# ================= Logistic Regression Test (sklearn) =================
'''
print ("Training Logistic Regression...")

# Initialize a Logistic Regression classifier from sklearn
from sklearn.linear_model import LogisticRegression

# Fit LR to training set
b = LogisticRegression()
b = b.fit(train_data_features, train_labels)

# Predict labels on test set
test_data_features = vectorizer.transform(testReviews)
test_data_features = test_data_features.toarray()
result = b.predict(test_data_features)

# Writing to sample.csv
print("Writing to sample.csv")
output = pd.DataFrame( data = {"id": test_id, "labels":result})
output.to_csv("sample.csv", index=False)

# Predict Training Set accuracy...
train_data_features = vectorizer.fit_transform(allReviews)
train_data_features = train_data_features.toarray()
result = b.predict(train_data_features)

# Print accuracy for sklearn Logistic Regression
accuracy = 0
for i in range(25000):
    if result[i] == train_labels[i]:
        accuracy = accuracy + 1

accuracy = ( accuracy / 25000 ) * 100

print ("Training Set Accuracy: " + str(accuracy) + " %\n")
'''

# ================= Logistic Regression Test (ours) =================
print ("Training Logistic Regression...", end='\t\t')

# Set n - number of features in training set
n = train_data_features.shape[1]

# Initializing fitting parameters
init_theta = np.zeros((n + 1, 1))

# Append intercept term to our training features
X_1 = np.append( np.ones((train_data_features.shape[0], 1)), train_data_features, axis=1)


# Check initial Cost & Gradient
# Computing Cost
#cost = computeCost(init_theta, X_1, train_labels, 1)
#print ('Cost at initial theta (zeros): ' + str(cost))

# Computing Gradient
#grad = computeGrad(init_theta, X_1, train_labels)
#print ('Gradient at initial theta (zeros):' + str(grad))

# Optimizing fitting parameters
opt_theta = optimize.fmin_l_bfgs_b(computeCost, init_theta, fprime=computeGrad, args=(X_1, train_labels, 1), disp=1, maxiter=400)
print ("Done!")

# Predicting the training set
p = predict(opt_theta[0], X_1)

accuracy = 0
for i in range(25000):
    if p[i] == train_labels[i]:
        accuracy = accuracy + 1
accuracy = (accuracy / 25000) * 100

print ("Training Set Accuracy: " + str(accuracy) + " %")

# Predict labels on test set
test_data_features = vectorizer.transform(testReviews)
test_data_features = test_data_features.toarray()

X_1_test = np.append( np.ones((test_data_features.shape[0], 1)), test_data_features, axis=1)
result = predict(opt_theta[0], X_1_test)

result = result.tolist()

# Convert to something we can write to .csv
for i, val in enumerate(result):
    result[i] = math.floor(val[0])
    

# Writing to sample2.csv
output = pd.DataFrame( data = {"id": test_id, "labels":result})
output.to_csv("sample.csv", index=False)
