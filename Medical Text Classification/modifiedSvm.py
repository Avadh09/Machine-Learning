# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 21:25:38 2018

@author: avadh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 18:30:36 2018

@author: avadh
"""

# 0.7945%  F1 score

# Support Vector Machine

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
with open('train.dat',"r") as fn:
    content_train = fn.readlines()


label = [i[0] for i in content_train]
description= [j[1:] for j in content_train]

# removing tab from strings
description = [i.replace('\t','') for i in description]

label = map(int, label)

# creating a pandas dataframe 
df = pd.DataFrame(list(zip(label,description)),columns=['Labels','Description'] )

X= df.Description
y= df.Labels

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.10)

#FOR TEST DATA
with open('test.dat',"r") as fn:
    content_test = fn.readlines()

test_df=pd.DataFrame(list(content_test), columns=['Test'])
#
Y=test_df.Test

#Vectorizing the dataset

from sklearn.feature_extraction.text import TfidfVectorizer
vect= TfidfVectorizer()
X_train_dtm= vect.fit_transform(X)

# Transforming testing data into a document-term-matrix
X_test_dtm= vect.transform(Y) 

from sklearn.linear_model import SGDClassifier
clf_svm=SGDClassifier(loss='hinge', penalty='elasticnet', n_iter=50) #62.7885 for niter 5 , hinge, 63.15 niter 50, hinge, l1
#
%time clf_svm.fit(X_train_dtm, y)


# class pridiction for X_train_dtm
y_pred_class= clf_svm.predict(X_test_dtm)

## Calculating accuracy for class predictions
#from sklearn import metrics
#metrics.accuracy_score(y_test,y_pred_class)
#
#print "Classification report: \n", (classification_report(y_test, y_pred_class))
#print "F1 weighted averaging:",(f1_score(y_test, y_pred_class, average='micro'))

#from sklearn import metrics 
#metrics.accuracy_score(y_test, y_pred_class)
#
#from sklearn.metrics import classification_report
#from sklearn.metrics import f1_score
#print "Classification report: \n", (classification_report(y_test, y_pred_class))
#print "F1 weighted averaging:",(f1_score(y_test, y_pred_class, average='macro'))

f= open('modifiedSvmOut.txt','w+')
for i in range(len(y_pred_class)):
    f.write(str(y_pred_class[i] )+ '\n')
    
f.close()