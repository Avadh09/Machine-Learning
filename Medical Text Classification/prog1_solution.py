# -*- coding: utf-8 -*-
"""
Created on Sat Mar 03 18:02:26 2018

@author: avadh
"""
#for i in range(10):
#    print description[i]
#    
#type(description)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV


# Data Preprocessing
#dataset = pd.read_csv('train.dat',sep='\t')
#vals = dataset.loc[:,:].values 

with open('train.dat',"r") as fn:
    content_train = fn.readlines()

label = [i[0] for i in content_train]
description= [j[1:] for j in content_train]

# removing tab from strings
description = [i.replace('\t','') for i in description]

label = map(int, label)

# creating a pandas dataframe 
df = pd.DataFrame(list(zip(label,description)),columns=['Labels','Description'] )

#df.head(10)
#df.Labels.value_counts()

# defines input and output for the model
X= df.Description
y= df.Labels
# Spliting X and y for training and testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.115)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape) 

# Vectorizing the dataset

from sklearn.feature_extraction.text import TfidfVectorizer
vect=TfidfVectorizer( stop_words='english')
X_train_dtm= vect.fit_transform(X_train)

# Transforming testing data into a document-term-matrix
X_test_dtm= vect.transform(X_test) 

# Building and Evaluationg a MODEL

from sklearn.svm import LinearSVC
clf_svc= LinearSVC( )
clf_svc.fit(X_train_dtm, y_train)

y_pred_class= clf_svc.predict(X_test_dtm) 
# import and instantiate Multinomial Naive Bayes model
#
#f= open('output.txt','w+')
#for i in range(len(y_pred_class)):
#    f.write(str(y_pred_class[i] )+ '\n')
#    
#f.close()

# Calculating accuracy for class predictions
from sklearn import metrics 
metrics.accuracy_score(y_test, y_pred_class)

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
print ("Classification report: \n", (classification_report(y_test, y_pred_class)))
print ("F1 weighted averaging:",(f1_score(y_test, y_pred_class, average='macro')))



