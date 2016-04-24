# -*- coding: utf-8 -*-

from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import os, re

def cleanReview(path):
    data = []
    for f in os.listdir(path):
        filePath = os.path.join(path, f)
        with open(filePath, 'r', encoding ='utf-8') as theFile:
            data.append(theFile.read().lower())
    
    for i in range(len(data)):      
        review = data[i]
           
        # Remove HTML tags
        review = BeautifulSoup(review, "html.parser").get_text()

        # Remove non alphanumerics
        review = re.sub('[^a-zA-Z0-9]', ' ', review)

        # Tokenize
        tokens = review.split()
        
        # Remove stop words
        #stops = set(stopwords.words("english"))
        #tokens = [w for w in tokens if not w in stops]
        
        # Remove empty strings
        tokens = filter(None, tokens)
        
        review = ( " ".join(tokens))

        data[i] = review
    return data