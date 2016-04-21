# -*- coding: utf-8 -*-
import os, sys, io, re

def buildTokens(path):
        theFile = io.open(path, 'r', encoding='utf-8')
        review = theFile.read().lower()
        theFile.close()

        tokens = re.split("\s+", review)
        
        # Remove nonalphanumerics
        for i, word in enumerate(tokens):
                tokens[i] = re.sub('\W+', '', word)
                
        # Remove empty strings
        tokens = filter(None, tokens)
                
        # Remove duplicates
        tokens = list(set(tokens))

        return tokens

if __name__ == '__main__':
    x = str(sys.argv[1])
    tokens = buildTokens(x)
    for i, word in enumerate(tokens):
        tokens[i] = str(word)


    print str(tokens)