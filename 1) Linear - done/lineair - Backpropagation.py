# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:19:56 2019

@author: Tim
"""
# import necessary Python packages
import os
import numpy as np



def derive(x):

    var = 1 - (x*x)
    
    return var 

def test(var):


    outputs = np.tanh(np.dot(synaptic_weights,var))  # maintenant on va faire la somme des inputs * weight  
    
    return outputs



def get_data():
    data=np.genfromtxt("1in_linear.txt",delimiter="")
    
    shape = np.shape(data)

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
    
    for iteration in range(20000):
            
        input_layer = inputs
        
        outputs = np.tanh(np.dot(input_layer,synaptic_weights))  # maintenant on va faire la somme des inputs * weight  
    
        error = training_outputs - outputs
        
        adjustment = error * derive(outputs)
        
        synaptic_weights += 0.1 * np.dot(input_layer.T, adjustment)
        
    #print (synaptic_weights)



if __name__ == '__main__':
    inputs, outputs = get_data()   
    training(inputs,outputs)
    print(test(-1.))
