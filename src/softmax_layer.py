#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 20:58:03 2017

@author: aleksandardjuric
"""

import numpy as np

class SoftmaxLayer(object):
    
    def __init__(self, n_inputs, n_classes, learning_rate, reg_lambda):
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        
        self.neurons = {}
        self.neurons['weights'] = np.random.uniform(-1,1,(n_classes,n_inputs))
        self.neurons['bias'] = np.zeros(n_classes)
        self.previous_layer_activation = []
    
    def activate(self, input):
        self.previous_layer_activation = input
        # Input multiplied by neuron weights
        activation = np.dot(self.neurons['weights'], input) + self.neurons['bias']
        # Softmax computation
        dist = self.__softmax(activation)
        return dist
    
    
    def __softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0, keepdims=True)
    
    
    def loss(self, softmax_probs, target, eps=1e-15):
        idx = np.argmax(target)
        predicted = softmax_probs[idx]
        predicted = np.clip(predicted, eps, 1 - eps)
        return -1 * np.log(predicted)
        
    
    def backprop(self, probabilities, target):
        # Calculate error signal
        error_signal = probabilities - target
        # Calculate gradients
        dW = np.outer(error_signal, self.previous_layer_activation.T)
        db = np.sum(error_signal, axis = 0, keepdims = True)

        # Regularization
        dW += self.neurons['weights'] * self.reg_lambda
        
        # Gradient descent
        self.neurons['weights'] -= self.learning_rate * dW
        self.neurons['bias'] -= self.learning_rate * db
        
        # Propagate error signal to previous layer
        previous_layer_error = np.dot(self.neurons['weights'].T, error_signal)
        previous_layer_error[self.previous_layer_activation <= 0] = 0
        
        return previous_layer_error
    
    
    
    
    
    
    