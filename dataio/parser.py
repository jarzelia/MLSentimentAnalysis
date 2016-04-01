import os
import re

dataPos = []
dataNeg = []
wordsPos = {}
wordsNeg = {}

path = "../train/pos/"

for f in os.listdir(path):
        filePath = os.path.join(path, f)
        dataPos.append(open(filePath, 'r', encoding='utf-8').read())


path2 = "../train/neg/"
for f in os.listdir(path2):
        filePath = os.path.join(path2, f)
        dataNeg.append(open(filePath, 'r', encoding='utf-8').read())


# regex needs work or need to trim strings with ','
# losing about 1/2 of our strings
for review in dataPos:
        tokens = re.split('[^a-zA-Z0-9_]', review)
        
        for t in tokens:
                if t in wordsPos:
                        wordsPos[t] = wordsPos[t] + 1
                else:
                        wordsPos[t] = 1
                        
fOut = open('pos_data_raw', 'w', encoding='utf-8')

for k, v in wordsPos.items():
        fOut.write(k + " " + str(v) + '\n')

