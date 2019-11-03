#-- CHATBOT --
#imports
import numpy as np
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

#set lowest ERROR_THRESHOLD to return the response
ERROR_THRESHOLD = 0.25
context = {}
old_state = {}
next_state = {}

#init stemmer
stemmer = LancasterStemmer()

#un-pickle generated data
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
output_empty = [0] * len(classes)
train_x = data['train_x']
train_y = data['train_y']

#reset underlying graph data
tf.reset_default_graph()
#generate neural net
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

#get model
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
#load saved model
model.load('./model_tflearn')

#-- functions --
def clean_up_sentence(sentence):
    #tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    #stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    #return stemmed words
    return sentence_words

#return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    #tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    #bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return([bag])

#pick the correlated pattern
def classify(sentence, context):
    #generate probabilities from the model
    results = model.predict([list(bow(sentence, words)[0] + context)])[0]
    #filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    #sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    #return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    filter = list(output_empty)
    if userID in context.keys() and context[userID] != '':
        filter[classes.index(context[userID])] = 1
    results = classify(sentence, filter)
    #if we have a classification then find the matching intent tag
    if results:
        #loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                #find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # check if this intent is contextual and applies to this user's conversation
                    if userID in context.keys() and show_details: print('context: ' + context[userID])
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        #set old state for this intent if necessary
                        if 'this_state' in i:
                            if show_details: print ('this_state:', i['this_state'])
                            old_state[userID] = i['this_state']
                        #set next state for this intent if necessary
                        if 'next_state' in i:
                            if show_details: print ('next_state:', i['next_state'])
                            next_state[userID] = i['next_state']
                        #set context for this intent if necessary
                        if 'context_set' in i:
                            if show_details: print ('context:', i['context_set'])
                            context[userID] = i['context_set']
                        #a random response from the intent
                        return i

            results.pop(0)

def main():
    testID = 1
    if not(testID in context.keys()):
        context[testID] = 'greeting'

    wasListening = False
    readline = input(' > ')
    if readline != 'x':
        if readline.strip() != '':
            email = None

            """ PREPROCESSING """
            #check if bot was meant to be listening
            if testID in context.keys() and context[testID] == "listening": wasListening = True
            else: wasListening = False

            #try to extract user email address
            if testID in context.keys() and context[testID] == "new_requirements_valid" and "@" in readline:
                #get first part
                temp = readline[0:readline.find("@")]
                if temp.find(" ") != -1: last_space = len(temp) - temp[::-1].find(" ")
                else: last_space = 0
                email = temp[last_space:]
                #get last part
                temp = readline[readline.find("@") + 1:]
                if temp.find(" ") != -1: last_space = temp.find(" ")
                else: last_space = len(temp)
                email += '@' + temp[0:last_space]
            elif testID in context.keys() and context[testID] == "new_requirements_valid":
                email = "INV"

            """ ML RESPONSE """
            output = response(readline,testID)

            """ POSTPROCESSING """
            #listening
            if testID in context.keys() and context[testID] == "listening" and wasListening:
                if output != None and output['tag'] == "invalid": print(random.choice(output['responses']))
                elif output != None and output['tag'] == "heard":
                    print("-stop recording")
                    print("-save recording to the right spot")
                    context[testID] = next_state[testID]
                    next_state[testID] = ""
                    old_state[testID] = ""
                    for i in intents['intents']:
                        if i['tag'] == context[testID]:
                            print(random.choice(i['responses']))
                else: print("-record that...")
            #listening for user email - invalid
            elif email != None and email == "INV":
                print("-We need a valid email address to continue")
            #listening for user email - valid
            elif email != None:
                print("-record: " + email)
            #otherwise use the ML generated response
            else:
                print(random.choice(output['responses']))
        main()

main()
