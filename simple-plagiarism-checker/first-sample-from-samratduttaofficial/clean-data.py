from itertools import count
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
    return cleanedText

# Attempting to read a large file can lead to a crash
# if there is not enough memory for the entire file to be read in at once.
# Reading the file in chunks makes it possible to access very large files
# by reading in one part of the file at a time.

# numOfChunks = 0
# for chunk_df in pd.read_csv(f'{PATH}\\plagcheckfile.csv', chunksize=100, encoding='latin-1'):
#     # print(chunk_df)
#     numOfChunks += 1
# print(numOfChunks)
def getData():    
    df = pd.read_csv(f'{PATH}\\plagcheckfile.csv', encoding='latin-1')
    texts = df['content'].values.tolist()
    result = []
    for i in range(0, 20):
        textDictionary = {"tag": i}
        # Compare: textDictionary without cleaing texts 
        # textDictionary.update(txts=texts[i])
        texts_i = cleanText(texts[i])
        textDictionary.update(txts=texts_i)
        result.append(textDictionary)
    finalDictionary = {"intents": result}
    print(len(finalDictionary["intents"]))
    return finalDictionary

combinedDictionary = dict()
print('Getting Text Data')
combinedDictionary.update(getData())
print('Total len of dictionary', len(combinedDictionary))

print('Saving text data dictionary')
np.save(PATH + '\\textDictionary.npy', combinedDictionary)

with open(PATH + '\\file.txt', 'w') as file:
    file.write(json.dumps(combinedDictionary))