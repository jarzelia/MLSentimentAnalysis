from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from lrFunctions import computeCost, computeGrad
from predict import predict
from scipy import optimize

import pandas as pd
import numpy as np
import os
import re

dataPos = []
dataNeg = []
dataTest = []

words = []
numOfFeatures = 5000
tfidfFlag = False

path = "../train/pos/"
path2 = "../train/neg/"
path3 = "../test/"

test_id = list(range(11000))


print ("Reading files from directories...")
for f in os.listdir(path):
        filePath = os.path.join(path, f)
        with open(filePath, 'r', encoding ='utf-8') as theFile:
            dataPos.append(theFile.read().lower())
            
        #theFile = open(filePath, 'r', encoding='utf-8')
        #dataPos.append(theFile.read().lower())
        #theFile.close()
for f in os.listdir(path2):
        filePath = os.path.join(path2, f)
        with open(filePath, 'r', encoding ='utf-8') as theFile:
            dataNeg.append(theFile.read().lower())
        
for f in os.listdir(path3):
        filePath = os.path.join(path3, f)
        with open(filePath, 'r', encoding ='utf-8') as theFile:
            dataTest.append(theFile.read().lower())
print ("Done!\n")

print ("Cleaning reviews...")
print ("Cleaning positive training set...")
for i in range(len(dataPos)):      
        review = dataPos[i]
           
        # Remove HTML tags
        review = BeautifulSoup(review, "html.parser").get_text()

        # Remove non alphanumerics
        review = re.sub('[^a-zA-Z]', ' ', review)

        # Tokenize
        tokens = review.split()
        
        # Remove stop words
        stops = set(stopwords.words("english"))
        tokens = [w for w in tokens if not w in stops]
        
        # Remove empty strings
        tokens = filter(None, tokens)
        
        review = ( " ".join(tokens))

        dataPos[i] = review

print ("Cleaning negative training set...")
for i in range(len(dataNeg)):
        review = dataNeg[i]
           
        # Remove HTML tags
        review = BeautifulSoup(review, "html.parser").get_text()

        # Remove non alphanumerics
        review = re.sub('[^a-zA-Z]', ' ', review)

        # Tokenize
        tokens = review.split()
        
        # Remove stop words
        stops = set(stopwords.words("english"))
        tokens = [w for w in tokens if not w in stops]
        
        # Remove empty strings
        tokens = filter(None, tokens)

        review = ( " ".join(tokens))
        
        dataNeg[i] = review

print ("Cleaning test set...")
for i in range(len(dataTest)):
        if( (i+1)%1000 == 0 ):
            print( "Review %d of %d (Test)\n" % ( i+1, len(dataTest) ) )
        review = dataTest[i]
           
        # Remove HTML tags
        review = BeautifulSoup(review, "html.parser").get_text()

        # Remove non alphanumerics
        review = re.sub('[^a-zA-Z]', ' ', review)

        # Tokenize
        tokens = review.split()
        
        # Remove stop words
        stops = set(stopwords.words("english"))
        tokens = [w for w in tokens if not w in stops]
        
        # Remove empty strings
        tokens = filter(None, tokens)

        review = ( " ".join(tokens))
        
        dataTest[i] = review
print ("Done!\n")

allReviews = []
for review in dataPos:
    allReviews.append(review)
for review in dataNeg:
    allReviews.append(review)


# Create label (class) for training set
train_labels = []
for i in range(25000):
    if i > 12500:
        train_labels.append(0)
    else:
        train_labels.append(1)


print ("Creating bag of words ... \n")
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)

train_data_features = vectorizer.fit_transform(allReviews)
train_data_features = train_data_features.toarray()
'''
print ("Training the random forest...\n")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit forest to training set
forest = forest.fit( train_data_features, train_labels)

test_data_features = vectorizer.transform(dataTest)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

# Writing to sample.csv
'''

'''
output = pd.DataFrame( data = {"id": test_id, "labels":result})
output.to_csv("sample.csv", index=False)
'''
print ("Training logistic regression...\n")


# Initialize a Logistic Regression classifier
from sklearn.linear_model import LogisticRegression

# Fit LR to training set
b = LogisticRegression()
b = b.fit(train_data_features, train_labels)

test_data_features = vectorizer.transform(dataTest)
test_data_features = test_data_features.toarray()
result = b.predict(test_data_features)

# Writing to sample_lr.csv
output = pd.DataFrame( data = {"id": test_id, "labels":result})
output.to_csv("sample_lr.csv", index=False)


# Predict Training Set accuracy...
train_data_features = vectorizer.fit_transform(allReviews)
train_data_features = train_data_features.toarray()
result = b.predict(train_data_features)

# Print accuracy
accuracy = 0
for i in range(25000):
    if result[i] == train_labels[i]:
        accuracy = accuracy + 1

accuracy = ( accuracy / 25000 ) * 100

print ("Accuracy: " + str(accuracy) + " %")


# Testing our model

# Set n - number of features in training set
n = train_data_features.shape[1]

init_theta = np.zeros((n + 1, 1))

# Append intercept term to our training features
X_1 = np.append( np.ones((train_data_features.shape[0], 1)), train_data_features, axis=1)

cost = computeCost(init_theta, X_1, train_labels)
grad = computeGrad(init_theta, X_1, train_labels)

print ('Cost at initial theta (zeros):' + str(cost) + '\n')
print ('Gradient at initial theta (zeros):' + str(grad) + '\n')
opt_theta = optimize.fmin_bfgs(cost, init_theta, fprime = grad, args=(X_1, train_labels))

p = predict(opt_theta, train_data_features)

print(p)
