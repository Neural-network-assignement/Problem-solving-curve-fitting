import numpy as np
import matplotlib.pyplot as plt





 
def sig(x):
    
    return x





def derivative_function(x):
    
    return (1 - (x*x))





def get_data():

    data=np.genfromtxt("1in_tanh.txt",delimiter="")
    shape = np.shape(data)
    
    height = shape[1]
    width = shape[0]
    
    inputs = np.zeros((width,1))
    outputs = np.zeros((width,1))
    
    for i in range(width):
        inputs[i] = data[i][0]
        outputs[i] = data[i][1]
        
    return inputs,outputs




def init_variables():

    global weight1
    global weight2

    weights_layer_1 = np.random.normal(size=2)
    weights_layer_2 = np.random.normal(size=2)
    weights_layer_3 = np.random.normal(size=2)
    weight1 = np.vstack([weights_layer_1,weights_layer_2,weights_layer_3])
     
    weight2 = np.random.normal(size=2)
    
    bias_1 = np.zeros((2,1))
    bias_2 = 0
    


    return bias_1, bias_2






def training (inputs, outputs, bias_1, bias_2):
    global weight1
    global weight2
    
    learning_rate = 0.1
    


    #i = np.array([[1],[2]])
    #w = np.array([[2,2],[2,2]]) 
    #o = np.dot(w,i)
    #print(inputs[0][0])
    #print(inputs[0][1])

    tmp_inputs = np.zeros((2,1))
    
    inputs_tmp = inputs[0][0]
    
    print(inputs_tmp)


    hidden_layer = tmp_inputs*weight1
    print(hidden_layer)
    
"""
    hidden_layer[0][0]= np.tanh(hidden_layer[0][0]) 
    hidden_layer[1][0]= np.tanh(hidden_layer[1][0])
    #print(hidden_layer)
    
    
    output_layer = np.dot(weight2, hidden_layer) + bias_2
    output_layer = np.tanh(output_layer)
    #print(output_layer)
    
    
    error_output = np.abs(outputs[0] - output_layer)
    error_hidden_layer = np.zeros((2,1))
    #print(error_hidden_layer)
    error_hidden_layer[0][0] = (weight2[0]/(weight2[1] + weight2[0]))*error_output
    error_hidden_layer[1][0] = (weight2[1]/(weight2[1] + weight2[0]))*error_output
    
    adjustment_output = learning_rate * error_output * derivative_function(output_layer)
    adjustment_hidden = learning_rate * error_hidden_layer * derivative_function(hidden_layer)
    
    weight1 += np.dot( adjustment_hidden  , hidden_layer.T)
    weight2 += np.dot( adjustment_output , tmp_inputs.T)

"""        
        
 
    
def predict (my_input,bias_1, bias_2):
    
    
    tmp_input = np.zeros((2,1))
    tmp_input[0][0] = inputs[0][0]
    tmp_input[1][0] = inputs[0][1]
    
    hidden_layer = np.dot(weight1, tmp_input) + bias_1
    hidden_layer[0][0]= np.tanh(hidden_layer[0][0]) 
    hidden_layer[1][0]= np.tanh(hidden_layer[1][0])
    #print(hidden_layer)
    
    output_layer = np.dot(weight2, hidden_layer) + bias_2
    output_layer = np.tanh(output_layer) 
    print(output_layer)
    
    
    
    
    
if __name__ == '__main__':
    
    
    inputs,outputs = get_data()
    bias1,bias2 = init_variables()
    training(inputs,outputs,bias1,bias2)