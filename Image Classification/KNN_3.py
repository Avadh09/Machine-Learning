# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 23:11:59 2018

@author: avadh
"""


#%matplotlib inline
#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import NuSVC,NuSVR,LinearSVC,LinearSVR
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
#

X= np.loadtxt("train.dat")
y=np.loadtxt("train.labels")



#def normalize(arr):
#    arr/=arr.max()
#    #arr-=arr.mean()
#    
#    return arr

#X=normalize(X)
sc=StandardScaler()
X= sc.fit_transform(X)
#splitinf=g into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)
#
#X_test=normalize(X_test)
#Testing Data
#X_test=np.loadtxt("test.dat")
#X_test=normalize(X_test)


# scaling 
#data = StandardScaler().fit_transform(data)

#PCA
pca = PCA(n_components=300)

pca.fit(X)


#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


#plt.plot(var1)
svd = TruncatedSVD(n_components=40)#55 37
svd.fit(X)

X_train_svd = svd.transform(X_train)
X_test_svd = svd.transform(X_test)

## overSampling 
#from collections import Counter
##from sklearn.datasets import make_classification
#from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE
##from imblearn.over_sampling import SMOTE
##X, y = make_classification(n_classes=11, class_sep=11, weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
##                           n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
#print('Original dataset shape {}'.format(Counter(y)))
#ros = RandomOverSampler()
#X_res, y_res = ros.fit_sample(X_train_svd, y_train)
#
#print(sorted(Counter(y_res).items()))
#
#print('Resampled dataset shape {}'.format(Counter(y_res)))

#K-nearest neighbor Classifier
#X_res = normalize(X_res)
#X_test_svd = normalize(X_test_svd)
#clf = KNeighborsClassifier(n_neighbors=4, algorithm='kd_tree')


clf=AdaBoostClassifier(ExtraTreesClassifier(n_estimators=100),n_estimators=600, algorithm= 'SAMME', learning_rate=1.2)
clf.fit(X_train_svd, y_train)
y_pred = clf.predict(X_test_svd)
y_pred=y_pred.astype(int)

from sklearn import metrics 
score = metrics.accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
print ("Classification report: \n", (classification_report(y_test, y_pred)))
print ("F1 weighted averaging:",(f1_score(y_test, y_pred, average='micro')))

## Saving output to the file 
#f= open('RFoutput_ver3.txt','w+')
#for i in range(len(y_pred)):
#    f.write(str(y_pred[i] )+ '\n')
#    
#f.close()
