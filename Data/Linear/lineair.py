# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:19:56 2019

@author: Tim
"""
# import necessary Python packages
import os
import numpy as np



data=np.genfromtxt("1in_linear.txt",delimiter="")

shape = np.shape(data)

height = shape[1]
width = shape[0]

inputs = np.zeros((width,1))
outputs_training = np.zeros((width,1))

for i in range(width):
    inputs[i] = data[i][0]
    outputs_training[i] = data[i][1]


def sigmoid (x):                #on crée la fonction sigmoid de la prof
    return 1 /( 1 + np.exp(-x))

def sigmoid_deriv (x):
    return x * (1 - x)


def test(var):

    input_layer = var

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))  # maintenant on va faire la somme des inputs * weight  
    
    return outputs
    
    

synaptic_weights = np.random.random((1, 1)) # on randomise le poid de chaques synapses

print ('random starting synaptic weight')           # et on le print
print (synaptic_weights)

for iteration in range (20000):
        
    input_layer = inputs
    
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))  # maintenant on va faire la somme des inputs * weight  

    error = training_outputs - outputs
    
    adjustment = error * sigmoid_deriv(outputs)
    
    synaptic_weights += np.dot(input_layer.T, adjustment)
    
print (synaptic_weights)
print ('outputs after training')
print (outputs)

var = [1]
o = test(var)
print(-1.0)


