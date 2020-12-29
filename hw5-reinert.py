# Author: Jeremy Reinert
# Date : 3/6/2020
# Version: 1.0

"""Reads in Churn_Modelling.csv and builds a k-Nearest Neighbor Machine Learning Model"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# import file
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:12].values
y = dataset.iloc[:, 12].values

# Encoding categorical data
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# train the data
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X = X_train, y = y_train)

# test the model
predicted = knn.predict(X = X_test)
expected = y_test

wrong_predictions = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
 
# print accuracy
print('* * * * * * k-NN Model where k = 3 * * * * * *\n')
print('Accuracy for k-NN Model where K = 3')
print(f'Accuracy of the model: {knn.score(X_test, y_test):.2%}')

# confusion matrix
confusion = confusion_matrix(y_true = expected, y_pred = predicted)
print('Confusion matrix for k-NN Model where K = 3')
print(confusion)

# k-fold analysis
kfold = KFold(n_splits = 10, random_state = 11, shuffle = True)
scores = cross_val_score(estimator=knn, X=X, y=y, cv=kfold)

print(scores)
sum = 0
for i in scores:
    sum = sum + i

avg_accuracy = sum / 10
print('K-Fold Cross-Validation Accuracy where k = 10')
print(f'K-Fold Cross Validation Accuracy: {avg_accuracy:.2%}')

#k-NN Model where k = 1
# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# train the data
knn = KNeighborsClassifier(n_neighbors = 1) 
knn.fit(X = X_train, y = y_train)

# test the model
predicted = knn.predict(X = X_test)
expected = y_test

wrong_predictions = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
 
print('\n* * * * * * k-NN Model where k = 1 * * * * * *\n')
# print accuracy
print('Accuracy for k-NN Model where K = 1')
print(f'Accuracy of the model: {knn.score(X_test, y_test):.2%}')

# confusion matrix
confusion = confusion_matrix(y_true = expected, y_pred = predicted)
print('Confusion matrix for k-NN Model where K = 1')
print(confusion)

# k-fold analysis
kfold = KFold(n_splits = 10, random_state = 11, shuffle = True)
scores = cross_val_score(estimator=knn, X=X, y=y, cv=kfold)

print(scores)
sum = 0
for i in scores:
    sum = sum + i

avg_accuracy = sum / 10
print('K-Fold Cross-Validation Accuracy where k = 10')
print(f'K-Fold Cross Validation Accuracy: {avg_accuracy:.2%}')

#k-NN Model where k = 3
# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# train the data
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X = X_train, y = y_train)

# test the model
predicted = knn.predict(X = X_test)
expected = y_test

wrong_predictions = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
 
print('\n* * * * * * k-NN Model where k = 3 * * * * * *\n')
# print accuracy
print('Accuracy for k-NN Model where K = 3')
print(f'Accuracy of the model: {knn.score(X_test, y_test):.2%}')

# confusion matrix
confusion = confusion_matrix(y_true = expected, y_pred = predicted)
print('Confusion matrix for k-NN Model where K = 3')
print(confusion)

# k-fold analysis
kfold = KFold(n_splits = 10, random_state = 11, shuffle = True)
scores = cross_val_score(estimator=knn, X=X, y=y, cv=kfold)

print(scores)
sum = 0
for i in scores:
    sum = sum + i

avg_accuracy = sum / 10
print('K-Fold Cross-Validation Accuracy where k = 10')
print(f'K-Fold Cross Validation Accuracy: {avg_accuracy:.2%}')

#k-NN Model where k = 5
# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# train the data
knn = KNeighborsClassifier(n_neighbors = 5) 
knn.fit(X = X_train, y = y_train)

# test the model
predicted = knn.predict(X = X_test)
expected = y_test

wrong_predictions = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
 
print('\n* * * * * * k-NN Model where k = 5 * * * * * *\n')
# print accuracy
print('Accuracy for k-NN Model where K = 5')
print(f'Accuracy of the model: {knn.score(X_test, y_test):.2%}')

# confusion matrix
confusion = confusion_matrix(y_true = expected, y_pred = predicted)
print('Confusion matrix for k-NN Model where K = 5')
print(confusion)

# k-fold analysis
kfold = KFold(n_splits = 10, random_state = 11, shuffle = True)
scores = cross_val_score(estimator=knn, X=X, y=y, cv=kfold)

print(scores)
sum = 0
for i in scores:
    sum = sum + i

avg_accuracy = sum / 10
print('K-Fold Cross-Validation Accuracy where k = 10')
print(f'K-Fold Cross Validation Accuracy: {avg_accuracy:.2%}')

#k-NN Model where k = 7
# splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# train the data
knn = KNeighborsClassifier(n_neighbors = 7) 
knn.fit(X = X_train, y = y_train)

# test the model
predicted = knn.predict(X = X_test)
expected = y_test

wrong_predictions = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
 
print('\n* * * * * * k-NN Model where k = 7 * * * * * *\n')
# print accuracy
print('Accuracy for k-NN Model where K = 7')
print(f'Accuracy of the model: {knn.score(X_test, y_test):.2%}')

# confusion matrix
confusion = confusion_matrix(y_true = expected, y_pred = predicted)
print('Confusion matrix for k-NN Model where K = 7')
print(confusion)

# k-fold analysis
kfold = KFold(n_splits = 10, random_state = 11, shuffle = True)
scores = cross_val_score(estimator=knn, X=X, y=y, cv=kfold)

print(scores)
sum = 0
for i in scores:
    sum = sum + i

avg_accuracy = sum / 10
print('K-Fold Cross-Validation Accuracy where k = 10')
print(f'K-Fold Cross Validation Accuracy: {avg_accuracy:.2%}')


