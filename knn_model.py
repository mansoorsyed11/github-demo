# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 09:49:44 2022

@author: msyed
"""

import numpy as np 
import pandas as pd

training_data = pd.read_csv('C:/Users/msyed/Documents/MLOPs/MLOPs ML & DL Deployment/storepurchasedata.csv')

X = training_data.iloc[:, :-1].values
y = training_data.iloc[:,-1].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
# minkowski is for ecledian distance
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)



#from sklearn.neighbors import KNeighborsClassifier

#classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)



model = classifier.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score, classification_report

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


new_predition = model.predict(sc.transform(np.array([[40, 50000]])))

new_predition2 = model.predict(sc.transform(np.array([[40, 20000]])))#

new_predition3 = model.predict(sc.transform(np.array([[20, 20000]])))



import pickle

model_file = "knnmodel.pickle"

pickle.dump(model, open(model_file,'wb'))

scaler_file = "sc.pickle"

pickle.dump(sc, open(scaler_file,'wb'))