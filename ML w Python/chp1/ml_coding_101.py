#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:10:05 2022
@author: itzp32g
Initializing an ML dataset
"""

import numpy as np
#Create a multidimensional array. 
#All members must be of the same type.
x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))
print()


"""
scipy provides,among other functionality, advanced linear algebra routines, mathematical function
optimization, signal processing, special mathematical functions, and statistical distributions.
"""
from scipy import sparse
# Create a 2D NumPy array with a diagonal of ones, and zeros everywhere else
#Sparse matrices are used whenever we want to store a 2D array that contains mostly zeros
eye = np.eye(4)
print("NumPy array:\n{}".format(eye))
print()

# Convert the NumPy array to a SciPy sparse matrix in CSR format
# Only the nonzero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))
print()


"""
It provides functions
for making publication-quality visualizations such as line charts, histograms, scatter
plots, and so on.
"""
import matplotlib.pyplot as plt
# Generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# Create a second array using sine
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker="x")
plt.show()


"""
pandas DataFrame is a table, similar to an Excel spreadsheet.
"""
import pandas as pd
# create a simple dataset of people
data = {'Name': ["John", "Anna", "Peter", "Linda"],
'Location' : ["New York", "Paris", "Berlin", "London"],
'Age' : [24, 13, 53, 33]
}
data_pandas = pd.DataFrame(data)
# IPython.display allows "pretty printing" of dataframes
# in the Jupyter notebook
print(data_pandas)
print()


# Select all rows that have an age column greater than 30
print(data_pandas[data_pandas.Age > 30])
print()

import mglearn