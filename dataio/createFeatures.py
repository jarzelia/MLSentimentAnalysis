from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import pandas as pd
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


print ("Reading files from directories...\n")
for f in os.listdir(path):
        filePath = os.path.join(path, f)
        theFile = open(filePath, 'r', encoding='utf-8')
        dataPos.append(theFile.read().lower())
        theFile.close()

for f in os.listdir(path2):
        filePath = os.path.join(path2, f)
        theFile = open(filePath, 'r', encoding='utf-8')
        dataNeg.append(theFile.read().lower())
        theFile.close()
        
for f in os.listdir(path3):
        filePath = os.path.join(path3, f)
        theFile = open(filePath, 'r', encoding='utf-8')
        dataTest.append(theFile.read().lower())
        theFile.close()


print ("Cleaning reviews...\n")
for i in range(len(dataPos)):
        if( (i+1)%1000 == 0 ):
            print( "Review %d of %d (Positive)\n" % ( i+1, len(dataPos) ) )
            
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

for i in range(len(dataNeg)):
        if( (i+1)%1000 == 0 ):
            print( "Review %d of %d (Negative)\n" % ( i+1, len(dataNeg) ) )
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


allReviews = []
for review in dataPos:
    allReviews.append(review)
for review in dataNeg:
    allReviews.append(review)


# Create labels
labels = []
for i in range(25000):
    if i > 12500:
        labels.append(0)
    else:
        labels.append(1)


print ("Creating bag of words ... \n")
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer= "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)

train_data_features = vectorizer.fit_transform(allReviews)
train_data_features = train_data_features.toarray()

print ("Training the random forest...\n")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit forest to training set
forest = forest.fit( train_data_features, labels)

test_data_features = vectorizer.transform(dataTest)
test_data_features = test_data_features.toarray()
result = forest.predict(test_data_features)

# Writing to sample.csv

id_labels = list(range(11000))

output = pd.DataFrame( data = {"id": id_labels, "labels":result})

output.to_csv( "sample.csv", index=False)

