# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:39:49 2019

@author: Tim
"""
 
import numpy as np
# This implementation of a standard feed forward network (FFN) is short and efficient, 
# using numpy's array multiplications for fast forward and backward passes. The source 
# code comes with a little example, where the network learns the XOR problem.
#
# Copyright 2008 - Thomas Rueckstiess
def sigmoid(x):
    
    return 1 / (1 + np.exp(-x))

def derivative(x):
    
    return x*(1-x)

 
###################################################################################################

def get_data():
    
    global width

    data=np.genfromtxt("2in_xor.txt",delimiter="")
    shape = np.shape(data)
    
    width = shape[0]
    
    inputs = np.zeros((width,2))
    outputs = np.zeros((width,1))
    
    for i in range(width):
        inputs[i][0] = data[i][0]
        inputs[i][1] = data[i][1]
        outputs[i] = data[i][1]
        
    return width,inputs,outputs


###################################################################################################


     
def feedforward(nIn, nHidden, nOut, inputs, teach):
    
    

    # learning rate
    alpha = 0.1
    
    global hWeights,oWeights,hActivation,oActivation,iOutput,oOutput,hOutput
                                             
    
     
    # initialize weights randomly (+1 for bias)
    hWeights = np.random.random((nHidden, nIn+1)) 
    oWeights = np.random.random((nOut, nHidden+1))
    #○print(oWeights.shape)
    #print(" ")
    #print(hWeights.shape)
     
    # activations of neurons (sum of inputs)
    hActivation = np.zeros((nHidden, 1), dtype=float)
    oActivation = np.zeros((nOut, 1), dtype=float)
     
    # outputs of neurons (after sigmoid function)
    iOutput = np.zeros((nIn+1, 1), dtype=float)      # +1 for bias
    hOutput = np.zeros((nHidden+1, 1), dtype=float)  # +1 for bias
    oOutput = np.zeros((nOut), dtype=float)
     
    # deltas for hidden and output layer
    hDelta = np.zeros((nHidden), dtype=float)
    oDelta = np.zeros((nOut), dtype=float)   
 


    
    # set input as output of first layer (bias neuron = 1.0)
    iOutput[:-1, 0] = inputs
    iOutput[-1:, 0] = 0
     
    # hidden layer
    hActivation = np.dot(hWeights, iOutput)
    hOutput[:-1, :] = sigmoid(hActivation)
    #○print(self.hOutput) 
    # set bias neuron in hidden layer to 1.0
    hOutput[-1:, :] = 0
     
    # output layer
    oActivation = np.dot(oWeights, hOutput)
    oOutput = sigmoid(oActivation)
 
    error = oOutput - teach 
    
    #error_total = 0.5*(oOutput - teach)**2
    #print(error_total)
     
    
    
    # deltas of output neurons
    oDelta = derivative(oActivation) * error        
    # deltas of hidden neurons
    hDelta = derivative(hActivation) * np.dot(oWeights[:,:-1].transpose(), oDelta)

    # apply weight changes
    hWeights = hWeights - alpha * np.dot(hDelta, iOutput.transpose()) 
    oWeights = oWeights - alpha * np.dot(oDelta, hOutput.transpose())
 
    
    
#################################################################################################   


def test(var):

 
    # set input as output of first layer (bias neuron = 1.0)
    iOutput[:-1, 0] = var
    iOutput[-1:, 0] = 0
    #print(var)
    # hidden layer
    hActivation = np.dot(hWeights, iOutput)
    hOutput[:-1, :] = sigmoid(hActivation)
    #print(hActivation)
    #○print(self.hOutput) 
    # set bias neuron in hidden layer to 1.0
    hOutput[-1:, :] = 0
     
    # output layer
    oActivation = np.dot(oWeights, hOutput)
    oOutput = sigmoid(oActivation)
    print(oOutput[0][0])


 
if __name__ == '__main__':
    ''' 
    XOR test example for usage of ffn
    '''
     
    width,inputs,output = get_data()
    
    # define training set
     
    # create network

    for a in range(2000):
        for i in range(width):
            feedforward(2,9,1,inputs[i],output[i])
    test([0,0])
    
