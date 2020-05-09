# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:19:56 2019

@author: Tim
"""
# import necessary Python packages
import os
import numpy as np
from random import *
import matplotlib as mpl
import matplotlib.pyplot as plt


def derive(x):

    var = 1 - (x*x)
    
    return var 

def test(var):


    outputs = np.tanh(np.dot(synaptic_weights,var))  # maintenant on va faire la somme des inputs * weight  
    
    return outputs



def get_data():
    data=np.genfromtxt("1in_linear.txt",delimiter="")
    
    shape = np.shape(data)
    
    global width

    width = shape[0]
    
    inputs = np.zeros((width,1))
    training_outputs = np.zeros((width,1))
    
    for i in range(width):
        
        inputs[i]           = data[i][0]
        training_outputs[i] = data[i][1]
        
    return inputs,training_outputs



    
def training(inputs,training_outputs):
    
    global synaptic_weights

    synaptic_weights = np.random.random((1, 1)) # on randomise le poid de chaques synapses
    
    #print ('random starting synaptic weight')           # et on le print
    #print (synaptic_weights)
    mean_sqrt_error = 10
    epoch = 0
    while mean_sqrt_error > 0.009:
        
        epoch += 1
            
        input_layer = inputs
        
        outputs = np.tanh(np.dot(input_layer,synaptic_weights))  # maintenant on va faire la somme des inputs * weight  
    
        error = training_outputs - outputs
        
        mean_sqrt_error = 0
        for i in range(width):
            
            mean_sqrt_error += error[i]**2
            
        mean_sqrt_error = mean_sqrt_error/width 
        #print(mean_sqrt_error)
        
        adjustment = error * derive(outputs)
        
        synaptic_weights += 0.1 * np.dot(input_layer.T, adjustment)
        
    return epoch,synaptic_weights


def test(inputs,outputs,weight):

        
    input_layer = inputs
    
    outputs_new = np.tanh(np.dot(input_layer,weight))  # maintenant on va faire la somme des inputs * weight  

    plt.plot(outputs,"g")
    plt.plot(outputs_new,"r")

if __name__ == '__main__':
    inputs, outputs = get_data()
    
    samples = 1000
    
    epoch_average = 0
    
    epoch, weight = training(inputs,outputs)
    print(epoch)
    
    test(inputs,outputs,weight)
    """
    for i in range(samples):
        
        epoch_average += training(inputs,outputs)
    
    epoch_average = epoch_average / samples
    
    print(epoch_average)
    """
    