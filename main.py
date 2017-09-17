#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 22:10:18 2017

@author: aleksandardjuric
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import pandas as pd

from sklearn import model_selection, preprocessing
from cnn import CNN

# Loading training data
dataframe = pd.read_csv('mnist_train.csv')
X_train = dataframe.values[:,1:]
X_train = X_train / 255.0
y_train = dataframe.values[:,0]

print("Training data loaded.")

#np.random.seed(0)
#X_train, y_train = datasets.make_moons(1000, noise=0.20)
#plt.scatter(X_train[:,0], X_train[:,1], s=40, c=y_train, cmap=plt.cm.Spectral)

# Hyperparameters
step_size = 5e-3
reg_strength = 5e-3
n_iter = 5
n_layers = 2
neurons_per_layer = 784

# Network initialization
model = CNN(step_size, reg_strength, n_iter, n_layers, neurons_per_layer)

# Training
model.fit(X_train, y_train)

# Testing
dataframe = pd.read_csv('mnist_test.csv')
X_test = dataframe.values[:,1:]
X_test = X_test / 255.0
y_test = dataframe.values[:,0]
print("Testing data loaded.")

#np.random.seed(0)
#X_test, y_test = datasets.make_moons(200, noise=0.20)
#plt.scatter(X_test[:,0], X_test[:,1], s=40, c=y_test, cmap=plt.cm.Spectral)

print("Accuracy: %02f" % (model.test_accuracy(X_test, y_test)))






