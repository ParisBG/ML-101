#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:35:03 2022
@author: itzp32g
Sample Iris ML model
"""

from sklearn.datasets import load_iris

iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print()


'''
The value of the key DESCR is a short description of the dataset.
'''
print(iris_dataset['DESCR'][:193] + "\n...")
print()


'''
The value of the key target_names is an array of strings, containing the species of
flower that we want to predict:
'''
print("Target names: {}".format(iris_dataset['target_names']))
print()


'''
The value of feature_names is a list of strings, giving the description of each feature:
'''
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print()


'''
The data itself is contained in the target and data fields.
'''
print("Type of data: {}".format(type(iris_dataset['data'])))
print()


'''
The rows in the data array correspond to flowers, while the columns represent the
four measurements that were taken for each flower. The shape of the data array is the number of samples multiplied by
the number of features.
'''
print("Shape of data: {}".format(iris_dataset['data'].shape))
print()


'''
Here are the feature values for the first five
samples
'''
print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
print()


'''
The target array contains the species of each of the flowers that were measured, also
as a NumPy array
'''
print("Type of target: {}".format(type(iris_dataset['target'])))
print()


'''
The species are encoded as integers from 0 to 2:
'''
#The meanings of the numbers are given by the iris['target_names'] array: 0 means setosa, 1 means versicolor, and 2 means virginica.
print("Target:\n{}".format(iris_dataset['target']))
print()


'''
scikit-learn contains a function that shuffles the dataset and splits it for you: the
train_test_split function. This function extracts 75% of the rows in the data as the
training set, together with the corresponding labels for this data. The remaining 25%
of the data, together with the remaining labels, is declared as the test set.
'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)



'''
The output of the train_test_split function is X_train, X_test, y_train, and
y_test, which are all NumPy arrays. X_train contains 75% of the rows of the dataset,
and X_test contains the remaining 25%:
'''
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
print()


'''
The data points are colored
according to the species the iris belongs to. To create the plot, we first convert the
NumPy array into a pandas DataFrame. pandas has a function to create pair plots
called scatter_matrix. The diagonal of this matrix is filled with histograms of each
feature
'''
import pandas as pd
from pandas.plotting import scatter_matrix
import mglearn
# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)



'''
Now we can start building the actual machine learning model. There are many classification
algorithms in scikit-learn that we could use. Here we will use a k-nearest
neighbors classifier

To make a prediction for a new data point, the algorithm
finds the point in the training set that is closest to the new point. Then it assigns the
label of this training point to the new data point.

The k in k-nearest neighbors signifies that instead of using only the closest neighbor
to the new data point, we can consider any fixed number k of neighbors in the training
(for example, the closest three or five neighbors).

'''
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)



'''
To build the model on the training set, we call the fit method of the knn object,
which takes as arguments the NumPy array X_train containing the training data and
the NumPy array y_train of the corresponding training labels:
'''
knn.fit(X_train, y_train)


import numpy as np

'''
We can now make predictions using this model on new data for which we might not
know the correct labels. Imagine we found an iris in the wild with a sepal length of
5 cm, a sepal width of 2.9 cm, a petal length of 1 cm, and a petal width of 0.2 cm.
What species of iris would this be? We can put this data into a NumPy array, again by
calculating the shape—that is, the number of samples (1) multiplied by the number of
features (4)
'''
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))
print()


'''
To make a prediction, we call the predict method of the knn object:
'''
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
iris_dataset['target_names'][prediction]))
print()


'''
We can measure how well the model works by
computing the accuracy, which is the fraction of flowers for which the right species
was predicted:
'''
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print()

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
'''
We can also use the score method of the knn object, which will compute the test set
accuracy for us:
'''
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


#THIS is the core machine learning algorithm code
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))