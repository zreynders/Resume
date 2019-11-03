#-- CHATBOT TEACHER --
#imports
import numpy as np
import pandas as pd
import random
import tflearn
import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle

#import json intents
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

#declare vars
words = []
classes = []
documents = []
ignore_words = ['?']
stemmer = LancasterStemmer()

#deconstruct intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word in the pattern
        w = nltk.word_tokenize(pattern)
        #add words to word list
        words.extend(w)
        #add the the words to documents with the tag
        if 'context_filter' in intent:
            documents.append((w, intent['context_filter'], intent['tag']))
        else:
            documents.append((w, '', intent['tag']))
        #add the tag to classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#stem each word
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]

#remove duplicates words
words = sorted(list(set(words)))

#remove duplicate classes
classes = sorted(list(set(classes)))

#declare vars
training = []
output = []
output_empty = [0] * len(classes)

#generate a bag of words for each pattern
for doc in documents:
    bag = []
    #get list of words for pattern
    pattern_words = doc[0]
    #get the stem of each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    #create bag of words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    #set current context_filter to 1
    filter = list(output_empty)
    if doc[1] != '':
        filter[classes.index(doc[1])] = 1

    #set current tag to 1
    output_row = list(output_empty)
    output_row[classes.index(doc[2])] = 1

    training.append([bag, filter, output_row])

#shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

#split train and test data
train_x = list(training[:,0] + training[:,1])
train_y = list(training[:,2])

#reset underlying graph data
tf.reset_default_graph()

#build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

#define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

#start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)

#save the model
model.save('model_tflearn')

#pickle other generated data
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )
