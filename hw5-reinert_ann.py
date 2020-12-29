# Author: Jeremy Reinert
# Date : 2/18/2020
# Version: 1.0

"""Reads in Churn_Modelling.csv and builds an Artificial Neural Network and k-Nearest Neighbor Machine Learning Models"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import time

# 1 PROCESS DATA
#import file
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:12].values
y = dataset.iloc[:, 12].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 2 MAKE THE ARTIFICIAL ANN
# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))

# Adding the second hidden layer
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
print('\nRUNNING EPOCHS')
start = time.time()
classifier.fit(X_train, y_train, batch_size = 100, epochs = 100)
end = time.time()
run_time = end - start
print('\nTIME ANALYSIS')
print(f'Time: {run_time}')

# 3 MAKE PREDICTIONS
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
print('\nCONFUSION MATRIX')
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Print out test data accuracy
print('\nACCURACY ANALYSIS WITH TEST DATA')
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print(f'Accuracy of the test data: {accuracy:.2%}')

#Possibility of client leaving
print('\nPREDICTING IF A CLIENT WILL LEAVE')
x = np.array([[550, 0, 32, 4, 1000, 3, 1, 0, 45000]])
new_pred = classifier.predict(sc.transform(x))

print(f'The chance of the customer leaving is: {new_pred[0][0]:.2%}')

new_pred2 = (new_pred > .5)
print(f'The customer will leave: {new_pred2[0][0]}')

