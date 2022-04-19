from matplotlib.pyplot import text
import pandas as pd
import numpy as np
import re
import json
import os

PATH = os.path.dirname(os.path.realpath(__file__))

def cleanText(Text):
    cleanedText = Text.replace('\n', ' ').lower()
    cleanedText = cleanedText.replace("\xc2\xa0", "")
    cleanedText = re.sub('([,])', '', cleanedText)
    cleanedText = re.sub(' +', ' ', cleanedText)
    cleanedText = cleanedText.encode('ascii', 'ignore').decode('ascii')
    return cleanText

def getData():
    df = pd.read_csv(f'{PATH}\\plagcheckfile.csv')
    texts = df['content'].values.tolist()
    result = []
    for i in range(0,4):
        textDictionary = {"tag":i}
        texts_i = cleanText(texts[i])
        print(texts_i)
        textDictionary.update(txts = texts_i)
        result.append(textDictionary)
    finalDictionary = { "intents" : result }   
    return finalDictionary


combinedDictionary = dict()
print ('Getting Text Data')
combinedDictionary.update(getData())
print ('Total len of dictionary', len(combinedDictionary))

print ('Saving text data dictionary')
np.save(PATH + '\\textDictionary.npy', combinedDictionary)

with open(PATH + '\\file.txt', 'w') as file:
     file.write(json.dumps(combinedDictionary))
