#Main Code
import numpy as np
import pandas as pd
import re
import pickle
from sklearn import preprocessing, model_selection, svm
from datetime import datetime as time
import datetime
import sys
import os

#---FUNCTIONS---
#split a string into tokens
def splitstring(string):
    string = re.sub('[^a-zA-Z0-9 \n\.]',' ',string)
    string = string.split()
    string2 = string[:]
    for token in string2:
        for token2 in string2:
            if len(token2) > 1 and len(token) > 1:
                if token2 != token and token2 in token:
                    string.append(token.replace(token2, ""))
    return string

#get the ord of a char ignoring Null values
def nonan_ord(string):
    if type(string) == type(-1):
        return string
    else:
        string = ord(string)
        return string

#get the char of an ord
def nonan_inverse_ord(string):
    string = chr(string)
    return string

#split text into a char list
def splitcolumn(string):
    string = list(string)
    return string

def main_start(string, save):
    timesince = time.now()
    
    this_dir = os.path.dirname(os.path.realpath(__file__))

    #---UNPICKLE---
    #import ML package
    clf = pickle.load(open(this_dir + "\data\BSTokenLabeler_clf.pickle","rb"))
    le_label = pickle.load(open(this_dir + "\data\BSTokenLabeler_le_label.pickle","rb"))
    onehote_feature = pickle.load(open(this_dir + "\data\BSTokenLabeler_onehote_feature.pickle","rb"))
        
    #---PREDICTION---
    line_df = pd.DataFrame([],columns=['TOKEN', 'LABEL'])
    for token in splitstring(string):
        if token.lower() != "deposit":
            #run ML package on token
            string = splitcolumn(token)
            string = pd.DataFrame([string])
            
            #add needed columns
            for i in range(0,26):
                if not string.columns.contains(i):
                    string[i] = 999
            
            #replace chars with ascii code
            for column in string.columns:
                string[column] = string[column].apply(nonan_ord)
            
            #encode label
            string_column_count = len(string.columns)
            for cid in string.columns:
                column = np.array(string[cid])
                column = column.reshape(len(column), 1)
                features_onehot = onehote_feature.transform(column)
                column = pd.DataFrame(features_onehot)
                column = column.add_suffix('_'+str(cid))
                string2 = string.drop(cid,1)
                string = string2.join(column)
                
            #make prediction
            prediction = le_label.inverse_transform(clf.predict(string))

            #convert prediction to array
            temp = pd.DataFrame([[token, prediction[0]]],columns=['TOKEN','LABEL'])
            line_df = line_df.append(temp, ignore_index=True)
    save_dir = save
    line_df.to_csv(save_dir, index=False, mode='a', header=False)
    return save_dir

print(main_start(sys.argv[1], sys.argv[2]))
