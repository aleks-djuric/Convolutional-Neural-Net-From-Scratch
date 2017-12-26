#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 22:10:18 2017

@author: aleksandardjuric
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import pandas as pd

from sklearn import model_selection, preprocessing
from cnn import CNN

def inputs():
    
    project_path = '/Main/Projects/CNNFromScratch'

    # Training data
    filepath = os.path.join(project_path, 'data/mnist_train.csv')
    dataframe = pd.read_csv(filepath)
    X_train = dataframe.values[:,1:]
    X_train = X_train / 255.0
    y_train = dataframe.values[:,0]
    print("Training data loaded.")
    
    # Testing data
    filepath = os.path.join(project_path, 'data/mnist_test.csv')
    dataframe = pd.read_csv(filepath)
    X_test = dataframe.values[:,1:]
    X_test = X_test / 255.0
    y_test = dataframe.values[:,0]
    print("Testing data loaded.")
    
#    np.random.seed(0)
#    X_train, y_train = datasets.make_moons(1000, noise=0.20)
#
#    np.random.seed(0)
#    X_test, y_test = datasets.make_moons(200, noise=0.20)
    
    return X_train, y_train, X_test, y_test


# Hyperparameters
step_size = 5e-3
reg_strength = 5e-3
n_iter = 5
n_layers = 2
neurons_per_layer = 784

## Hyperparameters
#step_size = 5e-3
#reg_strength = 5e-3
#n_iter = 3
#n_layers = 2
#neurons_per_layer = 10

# Network initialization
model = CNN(step_size, reg_strength, n_iter, n_layers, neurons_per_layer)

X_train, y_train, X_test, y_test = inputs()

# Training
model.fit(X_train, y_train)
# Testing
print("Accuracy: %02f" % (model.test_accuracy(X_test, y_test)))






