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

        self.classes = []
        self.one_hot = {}
        self.layers = []


    def __create_one_hot(self, classes):
        one_hot = {}
        for i,class_ in enumerate(classes):
            vector = np.zeros(len(classes))
            vector[i] = 1
            one_hot[class_] = vector
            
        return one_hot
            
    
    def __build_model(self, n_features, n_classes):
        # Create layers of network
        layers = []
        for l in range(self.n_layers):
            # First layer
            if(l == 0):
                layers.append(ReluLayer(n_features, self.neurons_per_layer,
                                             self.learning_rate, self.reg_lambda))
            # Last layer
            elif(l == self.n_layers - 1):
                layers.append(SoftmaxLayer(self.neurons_per_layer, n_classes,
                                                self.learning_rate, self.reg_lambda))
            # Middle layers
            else:
                layers.append(ReluLayer(self.neurons_per_layer, self.neurons_per_layer,
                                             self.learning_rate, self.reg_lambda))
                
        return layers
    
    
    def __train_model(self, inputs, targets):
        # Iterate through training data n_iter number of times
        for _ in range(self.n_iter):
            # Calculates loss after each iteration
            loss = 0
            for i, input_ in enumerate(inputs):
                # Forward propagation
                for layer in self.layers:
                    input_ = layer.activate(input_)

                # Resulting output probabilities of forward propagation
                output_probs = input_

                # Back-prop through last layer
                error_signal = self.layers[self.n_layers-1].backprop(output_probs, self.one_hot[targets[i]])

                # Back-prop through remaining layers
                for l in range(self.n_layers-2, 0, -1):
                    if(l == 0):
                        self.layers[l].backprop(error_signal, True)
                    else:
                        error_signal = self.layers[l].backprop(error_signal, False)
                # Aggregates loss from each sample
                loss += self.layers[self.n_layers-1].loss(output_probs, self.one_hot[targets[i]])

            # Reduces learning rate by 90% after each iteration
            for layer in self.layers:
                layer.learning_rate *= 0.1

            print(loss / len(inputs))
        
        
    def fit(self, inputs, targets):

        self.classes = np.unique(targets)
        self.one_hot = self.__create_one_hot(self.classes)
        
        n_classes = len(self.classes)
        n_features = len(inputs[0])
        self.layers = self.__build_model(n_features, n_classes)

        self.__train_model(inputs, targets)
        

    def predict(self, input_):
        # Forward propagation
        for layer in self.layers:
            input_ = layer.activate(input_)

        output_probs = input_

        # Finds output node with highest probability
        idx = np.argmax(output_probs, axis=None)

        # Matches node with corresponding class
        return self.classes[idx]


    def test_accuracy(self, inputs, targets):
        # Number of correct classifications
        n_correct = 0
        for i, input_ in enumerate(inputs):
            if(targets[i] == self.predict(input_)):
                n_correct += 1

        # % Accuracy
        return float(n_correct) / len(targets)
