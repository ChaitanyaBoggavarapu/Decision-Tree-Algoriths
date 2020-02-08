# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import tree,metrics


##To read  first column as index use index_col as zero
data = pd.read_csv('result_complete.csv',index_col=[0])

#To display Firest 5
data.head()

#To give info about File
data.info()



for i in range(1,data.shape[1],1):
    b =str(i)
    data[b],class_names = pd.factorize(data[b])
    
data['Type'],class_names = pd.factorize(data['Type'])

#dataFactorized = data.stack().rank(method='dense').unstack()
#To print Unique Data Types
print(data['Type'].unique())

##To select predict and target variable

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)

# train the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
dtree.fit(X_train, y_train)

# use the model to make predictions with the test data
y_pred = dtree.predict(X_test)
# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

