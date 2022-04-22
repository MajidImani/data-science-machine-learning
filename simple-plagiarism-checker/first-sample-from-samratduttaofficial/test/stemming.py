import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
porter = PorterStemmer()
words = ['Connects','Connecting','Connections','Connected','Connection','Connectings','Connect']
for word in words:
    print(word,"--->",porter.stem(word))
    
from nltk.stem import SnowballStemmer
snowball = SnowballStemmer(language='english')
words = ['preprocess','generate','generously','generation']
for word in words:
    print(word,"--->",snowball.stem(word))    


from nltk.stem import LancasterStemmer
lancaster = LancasterStemmer()
words = ['eating','eats','eaten','puts','tester']
for word in words:
    print(word,"--->",lancaster.stem(word))
    

from nltk.tokenize import word_tokenize
def stemming(text):
    list=[]
    for token in word_tokenize(text):
        list.append(lancaster.stem(token))
    return ' '.join(list)

sample = "In the example below, we constructed a function called stemming that uses word_tokenize to tokenize the text and then uses LancasterStemmer to stem down the token to its basic form."
print(word_tokenize(sample))
# print(stemming(text))