#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 18:17:55 2017

@author: aleksandardjuric
"""

import numpy as np

from relu_layer import ReluLayer
from softmax_layer import SoftmaxLayer

class CNN(object):
    
    def __init__(self, learning_rate, reg_lambda, n_iter, n_layers, neurons_per_layer):
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.n_iter = n_iter
        self.n_layers = n_layers
        self.neurons_per_layer = neurons_per_layer
        
        self.one_hot = {}
        self.classes = []
        self.layers = []
    
    
    def fit(self, inputs, targets):
        
        # Create one-hot vectors
        self.classes = np.unique(targets)
        for i,class_ in enumerate(self.classes):
            vector = np.zeros(len(self.classes))
            vector[i] = 1
            self.one_hot[class_] = vector 
            
        # Create layers of network
        for l in range(self.n_layers):
            # First layer
            if(l == 0):
                self.layers.append(ReluLayer(len(inputs[0]), self.neurons_per_layer,
                                             self.learning_rate, self.reg_lambda))
            # Last layer
            elif(l == self.n_layers - 1):
                self.layers.append(SoftmaxLayer(self.neurons_per_layer, len(self.classes),
                                                self.learning_rate, self.reg_lambda))
            # Middle layers
            else:
                self.layers.append(ReluLayer(self.neurons_per_layer, self.neurons_per_layer,
                                             self.learning_rate, self.reg_lambda))
        
        # Iterate through training data n_iter number of times
        for _ in range(self.n_iter):
            # Calculates loss after each iteration
            loss = 0
            for i,input in enumerate(inputs):
                
                # Forward propagation
                for layer in self.layers:
                    input = layer.activate(input)
                
                # Resulting output probabilities of forward propagation
                output_probs = input
                
                # Back-prop through last layer
                error_signal = self.layers[self.n_layers-1].backprop(output_probs,
                                          self.one_hot[targets[i]])
                
                # Back-prop through remaining layers
                for l in range(self.n_layers-2, 0, -1):
                    if(l == 0):
                        self.layers[l].backprop(error_signal, True)
                    else:
                        error_signal = self.layers[l].backprop(error_signal, False)
                # Aggregates loss from each input
                loss += self.layers[self.n_layers-1].loss(output_probs, self.one_hot[targets[i]])
            
            # Reduces learning rate by 90% after each iteration
            for layer in self.layers:
                layer.learning_rate *= 0.1
                
            print(loss / len(inputs))
            
            
    def predict(self, input):
        # Forward propagation
        for layer in self.layers:
            input = layer.activate(input)
        
        output_probs = input
        
        # Finds output node with highest probability
        idx = np.argmax(output_probs, axis=None)
        
        # Matches node with corresponding class
        return self.classes[idx]
    
    
    def test_accuracy(self, inputs, targets):
        # Number of correct classifications
        n_correct = 0
        for i,input in enumerate(inputs):
            if(targets[i] == self.predict(input)):
                n_correct += 1
        
        # % Accuracy
        return float(n_correct) / len(targets)
            
            
            
            
            
            
            
            
            
            