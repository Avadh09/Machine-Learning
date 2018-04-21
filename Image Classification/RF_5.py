
# 82.05 f1 score

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import RidgeClassifier

X= np.loadtxt("train.dat")
y=np.loadtxt("train.labels")


#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2)

#Testing Data
X_test=np.loadtxt("test.dat")

unique, counts = np.unique(y, return_counts=True)
cls_weight=dict(zip(unique, counts))

# splitinf=g into training and testing data

#RandomOverSampler_pipeline = make_pipeline_imb(RandomOverSampler(random_state=4), (random_state=42))
#smote_model = RandomOverSampler_pipeline.fit(X_train, y_train)
#RandomOverSampler_prediction = RandomOverSampler_pipeline_model.predict(X_test)

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

#
svd = TruncatedSVD(n_components=50)
svd.fit(X)

X_train_svd = svd.transform(X)
X_test_svd = svd.transform(X_test)

from collections import Counter
#from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE
#X, y = make_classification(n_classes=11, class_sep=11, weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
#                           n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
print('Original dataset shape {}'.format(Counter(y)))
ros = RandomOverSampler(ratio='all')
X_res, y_res = ros.fit_sample(X_train_svd, y)
print(sorted(Counter(y_res).items()))

print('Resampled dataset shape {}'.format(Counter(y_res)))
 
# training a linear SVM classifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

clf_5 = BaggingClassifier(base_estimator= RandomForestClassifier(n_estimators = 100, max_features = 'sqrt'),random_state= 7 )
#clf_5 = AdaBoostClassifier(base_estimator=clf_5, algorithm='SAMME', n_estimators= 600, learning_rate=0.8, random_state= 7 )
#RandomForestClassifier(n_estimators = 90, max_features = 'log2')
clf_5.fit(X_res,y_res)
# Predict on training set
y_pred = clf_5.predict(X_test_svd)
y_pred=y_pred.astype(int)


#score = metrics.accuracy_score(y_test, y_pred)
#
#from sklearn.metrics import classification_report
#from sklearn.metrics import f1_score
#print ("Classification report: \n", (classification_report(y_test, y_pred)))
#print ("F1 weighted averaging:",(f1_score(y_test, y_pred, average='micro')))

#Saving output to the file 
f= open('RF_5.txt','w+')
for i in range(len(y_pred)):
    f.write(str(y_pred[i] )+ '\n')
    
f.close()

#print (data[0])
#print (data)
#plt.imshow(data[1])
#from matplotlib import pyplot as plt
#plt.imshow(data[1], interpolation='nearest')
#plt.show()

# Normalizing the data 
#
#for i in range(len(data)):
#    data[i]=data[i]/np.linalg.norm(data[i])
#    
#print (data[0])