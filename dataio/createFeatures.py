from collections import Counter
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import os
import re

dataPos = []
dataNeg = []
words = []
numOfFeatures = 5000
tfidfFlag = False

path = "../train/pos/"
path2 = "../train/neg/"

w, h = 5001, 25000

Matrix = [[0 for x in range(w)] for y in range(h)] 

with open("theData.dat", 'r') as f:
    for line in f:
        tokens = line.split("\t")
        words.append(tokens[2].rstrip('\n'))

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
        
        # Remove empty strings
        tokens = filter(None, tokens)
                
        for t in tokens:
                if t in words:
                        Matrix[i][words.index(t)] += 1

for i in range(len(dataNeg)):
        if( (i+1)%1000 == 0 ):
            print( "Review %d of %d (Negative)\n" % ( i+1, len(dataNeg) ) )
        review = dataPos[i]
           
        # Remove HTML tags
        review = BeautifulSoup(review, "html.parser").get_text()

        # Remove non alphanumerics
        review = re.sub('[^a-zA-Z]', ' ', review)

        # Tokenize
        tokens = review.split()
        
        # Remove empty strings
        tokens = filter(None, tokens)
                
        for t in tokens:
                if t in words:
                        Matrix[i+12500][words.index(t)] += 1


fOut = open('trainingFeatures.dat', 'w')

for i in range(len(Matrix)):
    fOut.write(' '.join(map(str, Matrix[0])))
    fOut.write('\n')
fOut.close()
