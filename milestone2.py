import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from Encoder import *
from cleaning import*
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics, linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectKBest

data_for_clean=pd.read_csv('megastore-classification-dataset.csv')
######################################################DataCleaning
data=data_clean(data_for_clean)

#######################################################
X=data.iloc[:,:] #Features
Y=data['ReturnCategory'] #Label
##############################Encoding
cols2=('Ship Mode','Customer ID','Customer Name','Segment','Product Name','Main Category','Sub Category','City','Postal Code','Sales','Quantity','ReturnCategory','Duration','Main Category','Sub Category')

cols=('Ship Mode','Customer ID','Customer Name','Segment','Product Name','Main Category','Sub Category','City','ReturnCategory')
X= Feature_Encoder(X,cols)


#############################scaling
for i in  cols2 :
 X[i]= scale(X[i])
pd.set_option('display.max_columns', None)
print(X.head(5))

#########################################
#feature selection

top=feature_selection(X)

X=X[top]
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data[top].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

print("top features is",top)

#########################################################################
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.35,shuffle=True,random_state=100)

#########################################################GaussianNB

classifier_GaussianNB= GaussianNB()
classifier_GaussianNB.fit(X_train, y_train)
y_pred_GaussianNB = classifier_GaussianNB.predict(X_test)
print('GaussianNB: ',metrics.accuracy_score(y_test, y_pred_GaussianNB)*100)
###############################################################################DecisionTreeClassifier

Decision_tree=DecisionTreeClassifier(random_state=100,min_samples_split=10)
Decision_tree.fit(X_train,y_train)
y_p_Decision_tree=Decision_tree.predict(X_test)
print('DecisionTreeClassifier :',metrics.accuracy_score(y_test, y_p_Decision_tree)*100)

###################################################################################svm

model_svm = svm.SVC(kernel = 'rbf', random_state = 10000, C=1000000000)
model_svm.fit(X_train, y_train)
y_p_svm=model_svm.predict(X_test)
print('svm:',metrics.accuracy_score(y_test, y_p_svm)*100)

###################################################################################logistic

mod_logistic=LogisticRegression(solver='liblinear',C=10000000,random_state=100)
mod_logistic.fit(X_train,y_train)
y_p_logistic=mod_logistic.predict(X_test)
print('logistic :',metrics.accuracy_score(y_test, y_p_logistic)*100)

#################################################################################RandomForestClassifer

class_RandForest=RandomForestClassifier(random_state=100,min_samples_split=5,oob_score=False).fit(X_train,y_train)
y_rand=class_RandForest.predict(X_test)
print('RandomForestClassifer',metrics.accuracy_score(y_test, y_rand)*100)

######################################################################################AdaBoostClassifer

clf_model_dtree=AdaBoostClassifier(n_estimators=10,learning_rate=0.000000001,random_state=100)
clf_model_dtree=clf_model_dtree.fit(X_train,y_train)
clf_model_dtree_predict=clf_model_dtree.predict(X_test)
print('AdaBoostClassifer',metrics.accuracy_score(y_test, clf_model_dtree_predict)*100)
#######################################################################################################pickle
pickle_file_model=open('new_pickle.pickle','wb')
Encoder=LabelEncoder()
scale=MinMaxScaler()
feature_selection=SelectKBest()
pickle.dump(Encoder,pickle_file_model)
pickle.dump(scale,pickle_file_model)
pickle.dump(feature_selection,pickle_file_model)
pickle.dump(classifier_GaussianNB,pickle_file_model)
pickle.dump(Decision_tree,pickle_file_model)
pickle.dump(model_svm,pickle_file_model)
pickle.dump(mod_logistic,pickle_file_model)
pickle.dump(class_RandForest,pickle_file_model)
pickle.dump(clf_model_dtree,pickle_file_model)
