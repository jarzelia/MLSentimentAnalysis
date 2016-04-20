from collections import Counter
import os
import re

dataPos = []
dataNeg = []
wordsPos = {}
wordsNeg = {}

path = "../train/pos/"

for f in os.listdir(path):
        filePath = os.path.join(path, f)
        theFile = open(filePath, 'r', encoding='utf-8')
        dataPos.append(theFile.read().lower())
        theFile.close()

path2 = "../train/neg/"
for f in os.listdir(path2):
        filePath = os.path.join(path2, f)
        theFile = open(filePath, 'r', encoding='utf-8')
        dataNeg.append(theFile.read().lower())
        theFile.close()

for review in dataPos:
        tokens = re.split("\s+", review)

        # Remove duplicates
        tokens = list(set(tokens))     
        
        for t in tokens:
                # Remove non-alphanumeric characters
                t = re.sub('\W+', '', t)
                if t in wordsPos:
                        wordsPos[t] = wordsPos[t] + 1
                else:
                        wordsPos[t] = 1

                      
fOut = open('pos_data.dat', 'w')
for w in sorted(wordsPos, key = wordsPos.get, reverse=True)[:1000]:
        fOut.write(str(wordsPos[w]) + " " + w + "\n")
fOut.close()


for review in dataNeg:
        tokens = re.split("\s+", review)

        # Remove duplicates list->set->list
        tokens = list(set(tokens))
        
        for t in tokens:
                # Remove non-alphanumeric characters
                t = re.sub('\W+', '', t)
                if t in wordsNeg:
                        wordsNeg[t] = wordsNeg[t] + 1
                else:
                        wordsNeg[t] = 1

fOut2 = open('neg_data.dat', 'w')
for w in sorted(wordsNeg, key = wordsNeg.get, reverse=True)[:1000]:
        fOut2.write(str(wordsNeg[w]) + " " + w + "\n")
fOut2.close()

