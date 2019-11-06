import numpy as np
import matplotlib.pyplot as plt

# import necessary Python packages
import os



data=np.genfromtxt("2in_xor.txt",delimiter="")

shape = np.shape(data)

height = shape[1]
width = shape[0]

inputs = np.zeros((width,2))
outputs_training = np.zeros((width,1))

for i in range(width):
    inputs[i][0] = data[i][0]
    inputs[i][1] = data[i][1]
    outputs_training[i] = data[i][2]


def sigmoid (x):                #on crée la fonction sigmoid de la prof
    return 1 /( 1 + np.exp(-x))

def sigmoid_deriv (x):
    return x * (1 - x)

weight1=([0,0],[0,0])
weight2=([0,0])

def init_variables():
    
        #Init model variables (weights and bias)
    
    global weight1
    global weight2


    weights_layer_1 = np.random.normal(size=2)
    weights_layer_2 = np.random.normal(size=2)
    weight1 = np.vstack([weights_layer_1,weights_layer_2])
     
    weight2 = np.random.normal(size=2)
    
    bias_1 = np.zeros((2,1))
    bias_2 = 0

    return bias_1, bias_2

bias_1,bias_2 = init_variables()

learning_rate = 0.01

for a in range (200):
    for i in range (width):

        #i = np.array([[1],[2]])
        #w = np.array([[2,2],[2,2]]) 
        #o = np.dot(w,i)
        #print(inputs[0][0])
        #print(inputs[0][1])

        tmp_inputs = np.zeros((2,1))
        tmp_inputs[0][0] = inputs[i][0]
        tmp_inputs[1][0] = inputs[i][1]
        
        #print(tmp_inputs)
        #print(weight1)
        #print("")
    
        hidden_layer = np.dot(weight1, tmp_inputs) + bias_1
        #print(hidden_layer)
        hidden_layer[0][0]= sigmoid(hidden_layer[0][0])
        hidden_layer[1][0]= sigmoid(hidden_layer[1][0])
        #print(hidden_layer)
        
        
        output_layer = np.dot(weight2, hidden_layer) + bias_2
        output_layer = sigmoid(output_layer)
        #print(output_layer)
        
        
        error_output = outputs_training[i] - output_layer
        error_hidden_layer = np.zeros((2,1))
        #print(error_hidden_layer)
        error_hidden_layer[0][0] = (weight2[0]/(weight2[1] + weight2[0]))*error_output
        error_hidden_layer[1][0] = (weight2[1]/(weight2[1] + weight2[0]))*error_output
        
        adjustment_output = learning_rate * error_output * sigmoid_deriv(output_layer)
        adjustment_hidden = learning_rate * error_hidden_layer * sigmoid_deriv(hidden_layer)
        
        weight1 += np.dot(hidden_layer.T, adjustment_output)
        weight2 += np.dot(tmp_inputs.T, adjustement_hidden)
        
    i = 0
    
print (weight1)
print (weight2)

tmp_inputs[0][0] = 0
tmp_inputs[1][0] = 1

#print(tmp_inputs)
#print(weight1)
#print("")

hidden_layer = np.dot(weight1, tmp_inputs) + bias_1
#print(hidden_layer)
hidden_layer[0][0]= sigmoid(hidden_layer[0][0]) 
hidden_layer[1][0]= sigmoid(hidden_layer[1][0])
#print(hidden_layer)


output_layer = np.dot(weight2, hidden_layer) + bias_2
output_layer = sigmoid(output_layer)
print(output_layer)
