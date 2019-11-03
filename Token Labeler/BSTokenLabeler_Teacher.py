import numpy as np
import pandas as pd
import pymssql as pq
import re
import pickle
from sklearn import preprocessing, model_selection, svm
import os

#split string into tokens
def splitstring(string):
    string = re.sub('[^a-zA-Z0-9 \n\.]',' ',string)
    string = string.split()
    return string

#split text into a char list
def splitcolumn(string):
    string = list(str(string))
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

this_dir = os.path.dirname(os.path.realpath(__file__))

#---GET DATA---
#connect sql

#Queries removed for security reasons

#split custname into tokens
custname = custname.merge(custno,left_on='ENTITY_ID',right_on='CUST_PROS_NUMBER',suffixes=('_left','_right'))
custname = splitstring(custname['CUSTOMER_NAME'].to_string(header=False,index=False))
custname = pd.Series(custname)
custname = pd.DataFrame(custname,columns=['DATA'])
custname['LABEL'] = 'cna'


#rename custno columns
custno.columns = ['DATA','LABEL']

#Get classbook data
classbook = pd.read_csv(this_dir + "\data\classbook.csv", header=None, skip_blank_lines=False)
classbook.columns = ['DATA', 'LABEL']

#combine data
df = dealno.append(vinno).reset_index(drop=True)
df = df.append(custno).reset_index(drop=True)
df = df.append(custname).reset_index(drop=True)
#df = df.append(vinnoless).reset_index(drop=True)
df = df.append(classbook).reset_index(drop=True)

#split DATA column into char columns
df2 = df
temp = df2['DATA'].apply(splitcolumn).tolist()
temp = pd.DataFrame(temp)
temp.replace([None], value=999, inplace=True)
#add needed columns
for i in range(0,26):
    if not temp.columns.contains(i):
        temp[i] = 999

#replace chars with ascii code
for column in temp.columns:
    temp[column] = temp[column].apply(nonan_ord)

#merge data with original labels
df2 = temp.merge(df2['LABEL'],left_index=True,right_index=True)

#split X:y
X = df2.drop(['LABEL'], 1)
y = df2['LABEL']

#---LABEL ENCODING---
#init label encoder
le_label = preprocessing.LabelEncoder()
le_label.fit(y)

#encode label
label_encoded = le_label.transform(y)
y = pd.DataFrame(label_encoded)

#---FEATURE ENCODING---
#read charseries csv
charseries = pd.read_csv(this_dir + "\data\charseries.csv")

#init feature encoder
onehote_feature = preprocessing.OneHotEncoder(sparse=False)
onehote_feature.fit(charseries)

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000

#encode label
X_column_count = len(X.columns)
for cid in X.columns:
    column = np.array(X[cid])
    column = column.reshape(len(column), 1)
    features_onehot = onehote_feature.transform(column)
    column = pd.DataFrame(features_onehot)
    column = column.add_suffix('_'+str(cid))
    X2 = X.drop(cid,1)
    X = X2.join(column)


print(X)

#---MACHINE LEARNING---
#data split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

#teach
clf = svm.SVC(gamma='scale')
clf.fit(X_train,y_train)

#predict on test data
prediction = le_label.inverse_transform(clf.predict(X_test))
pred_df = pd.DataFrame(prediction, columns=['Prediction'])
temp = pd.DataFrame(le_label.inverse_transform(y_test), columns=['Actual'])
pred_df = pred_df.join(temp)
pred_df['Correct'] = (pred_df['Prediction']==pred_df['Actual'])

#decode X_test
X_test_decode = X_test.merge(df, 'left',left_index=True,right_index=True)
X_test_decode = X_test_decode[['LABEL', 'DATA']]
X_test_decode = X_test_decode.reset_index()
pred_df = X_test_decode.join(pred_df)
pred_df = pred_df[['DATA', 'LABEL', 'Prediction', 'Correct']]

#final output 4494
print('Full Prediction Table: ')
print(pred_df)
print('Incorrect Prediction Table: ')
print(pred_df.loc[pred_df['Correct'] == False])

#---SERIALIZATION---
#pickle data
pickle.dump(clf, open(this_dir + "\data\BSTokenLabeler_clf2.pickle","wb"))
pickle.dump(le_label, open(this_dir + "\data\BSTokenLabeler_le_label2.pickle","wb"))
pickle.dump(onehote_feature, open(this_dir + "\data\BSTokenLabeler_onehote_feature2.pickle","wb"))
