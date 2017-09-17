#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 18:03:22 2017

@author: aleksandardjuric
"""

import numpy as np

class ReluLayer(object):
    
    def __init__(self, n_features, n_neurons, learning_rate, reg_lambda):
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        
        self.neurons = {}
        self.neurons['weights'] = np.random.uniform(-1,1,(n_neurons,n_features))
        self.neurons['bias'] = np.zeros(n_neurons)
        self.previous_layer_activation = []

    
    def activate(self, input):
        self.previous_layer_activation = input
        # Input multiplied by neuron weights
        activation = np.dot(self.neurons['weights'], input) + self.neurons['bias']
        # Activation using ReLU transfer function
        activation = np.maximum(np.zeros(len(activation)), activation)
        return activation
    
    def backprop(self, error_signal, is_first_layer):
        # Calculate gradients
        dW = np.outer(error_signal, self.previous_layer_activation.T)
        db = np.sum(error_signal, axis = 0, keepdims = True)
        
        # Regularizaton
        dW += self.reg_lambda * self.neurons['weights']
        
        # Gradient descent
        self.neurons['weights'] -= self.learning_rate * dW
        self.neurons['bias'] -= self.learning_rate * db
        
        if is_first_layer:
            return
        else:
            # Propagate error signal to previous layer
            previous_layer_error = np.dot(self.neurons['weights'].T, error_signal)
            previous_layer_error[self.previous_layer_activation <= 0] = 0
        
            return previous_layer_error
        