from operator import le
import os
import json
import tflearn
import numpy
from nltk.stem.lancaster import LancasterStemmer
from socket import ntohl
import nltk
nltk.download('punkt')

stemmer = LancasterStemmer()

PATH = os.path.dirname(os.path.realpath(__file__))
with open(PATH + "\\file.txt") as file:
    dataset = json.load(file)

list_words = []
labels = []
docs_x = []  # List of all the question_patterns.
docs_y = []  # List of all the tags for specific Texts.

for intent in dataset["intents"]:
    # for pattern in intent["txts"]: #Stems the words. Finds the root of the word. Removes extra characters and symbols to find the root word.
    # Tokenizes the words. Breakes the words in the places of spaces and returns a list of all the words in it.
    split_words = nltk.word_tokenize(intent["txts"])
    # Using instead of looping and adding one word at a time in the list. It just extends the list untill all the words are in it.
    list_words.extend(split_words)
    # Adding the pattern of words inside docs_x list.
    docs_x.append(split_words)
    # For each pattern, it says what Tag it is a part of.
    docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])  # Adds all the tags in the labels list.

# Lower cases all the words in the list_words list.
list_words = [stemmer.stem(w.lower()) for w in list_words if w != "?"]
# Makes a set of the words to remove duplicate.
# This gives us the actual vocabulary size of the intent.
# Then converts it back to list and sorts it.
list_words = sorted(list(set(list_words)))
print(f'list_words: {list_words} \n labels: {labels}')

labels = sorted(labels)  # Sorts the labels where the tags are stored.
training = []  # contains the bag of words.
output = []  # The output list to choose the right tag for the output.

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    # Makes a list of all the words and tells if those words are occuring to find a pattern.
    bag = []
    # We are representing each sentence with a list the length of the amount of words in our models vocabulary.
    # Each position in the list will represent a word from our vocabulary.
    # If the position in the list is a 1 then that will mean that the word exists in our sentence, if it is a 0 then the word is nor present.

    # Stems all the words inside docs_x list.
    split_words = [stemmer.stem(w.lower()) for w in doc]

    for w in list_words:
        if w in split_words:
            # we are putting 1 in the bag of words list for the word (already present in the vocabulary list_words) present in the pattern and 0 if it is not.
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    # We are looking in the labels list to find were the tag is in that list. We
    # are making that value 1 in the output row.
    output_row[labels.index(docs_y[x])] = 1
    # We will create output lists which are the length of the amount of labels/tags we have in our dataset.
    # Each position in the list will represent one distinct label/tag, a 1 in any of those positions will show which label/tag is represented.

    # We are putting the bag of words in the training list.
    training.append(bag)
    # We are putting the output_row list in the output list.
    output.append(output_row)

# # bag_of_words function will transform our string input to a bag of words using our created words list
# def bag_of_words(s, list_words):
#     bag = [0 for _ in range(len(list_words))]

#     inp_str_words = nltk.word_tokenize(s)
#     inp_str_words = [stemmer.stem(word.lower()) for word in inp_str_words]

#     for search_element in inp_str_words:
#         for i, w in enumerate(list_words):
#             if w == search_element:
#                 bag[i] = 1

#     return numpy.array(bag)


# # we will convert our training data and output to numpy arrays.
# training = numpy.array(training)
# output = numpy.array(output)
# # tensorflow.reset_default_graph() #Resetting all the previous stuffs in the graph.

# # This finds the input shape that we are expecting for the model. Each training input is gonna be of the same length, so, 0.
# net = tflearn.input_data(shape=[None, len(training[0])])
# net = tflearn.fully_connected(net, 8)  # Hidden layer with 8 neurons.
# net = tflearn.fully_connected(net, 8)  # Another hidden layer with 8 neurons.
# net = tflearn.fully_connected(net, 8)  # Another hidden layer with 8 neurons.
# # length will be how many labels (how many tags) we have. Give us probability for each neuron in the network. The higest probability will be the output.
# net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
# net = tflearn.regression(net)

# model = tflearn.DNN(net)  # Model gets trained.

# try:
#     model.predict(PATH + "\\model.tflearn")
#     model.load()
# except:
#     # Fitting our data to the model. The number of epochs we set is the amount of times that the model will see the same information while training.
#     model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
#     # we can save it to the file model.tflearn for use in other scripts.
#     model.save(PATH + "\\model.tflearn")
