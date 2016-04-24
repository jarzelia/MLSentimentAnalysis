from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from lrFunctions import computeCost, computeGrad
from predict import predict
from scipy import optimize

import pandas as pd
import numpy as np
import os
import re
import math

dataPos = []        # Holds train/pos/ reviews
dataNeg = []        # Holds train/neg/ reviews
dataTest = []       # Holds test/ reviews

path = "../train/pos/"
path2 = "../train/neg/"
path3 = "../test/"

test_id = list(range(11000))


print ("Reading files from directories...", end='')
for f in os.listdir(path):
        filePath = os.path.join(path, f)
        with open(filePath, 'r', encoding ='utf-8') as theFile:
            dataPos.append(theFile.read().lower())
            
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
print ("Cleaning positive training set...", end='')
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
print ("Done!")
print ("Cleaning negative training set...", end='')
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
print ("Done!")
print ("Cleaning test set...", end='')
for i in range(len(dataTest)):
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

# Append dataPos and dataNeg together
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


print ("Creating bag of words ...", end='')
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)

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

test_data_features = vectorizer.transform(dataTest)
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
test_data_features = vectorizer.transform(dataTest)
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
print ("Training Logistic Regression...")

# Set n - number of features in training set
n = train_data_features.shape[1]

init_theta = np.zeros((n + 1, 1))

# Append intercept term to our training features
X_1 = np.append( np.ones((train_data_features.shape[0], 1)), train_data_features, axis=1)

# Computing Cost
#cost = computeCost(init_theta, X_1, train_labels, 1)
#print ('Cost at initial theta (zeros): ' + str(cost))

# Computing Gradient
#grad = computeGrad(init_theta, X_1, train_labels)
#print ('Gradient at initial theta (zeros):' + str(grad))

# Optimizing parameters theta
opt_theta = optimize.fmin_l_bfgs_b(computeCost, init_theta, fprime=computeGrad, args=(X_1, train_labels, 1), disp=1, maxiter=400)

# Predicting the training set
p = predict(opt_theta[0], X_1)

accuracy = 0
for i in range(25000):
    if p[i] == train_labels[i]:
        accuracy = accuracy + 1
accuracy = (accuracy / 25000) * 100

print ("Training Set Accuracy: " + str(accuracy) + " %\n")

# Predict labels on test set
test_data_features = vectorizer.transform(dataTest)
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
