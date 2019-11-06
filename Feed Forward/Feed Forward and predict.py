import numpy as np
import matplotlib.pyplot as plt

weight1=([0,0],[0,0])
weight2=([0,0])
 
def sig(x):
    
    return 

def derivative_function(x):
    
    return (1 - (x*x))

def database():

    """
    creation of the database
    """

    # definition of the number of exemples
    number_of_samples = 100
    #   creating sick persons in 2 types
    sick_type_1 = np.random.randn(number_of_samples, 2) + np.array([ -2 , 2 ])
    sick_type_2 = np.random.randn(number_of_samples, 2) + np.array([ 2 , -2 ])
    sick = np.vstack([sick_type_1,sick_type_2])
    #   creating healthy persons in 2 types
    healthy_type_1 = np.random.randn(number_of_samples, 2) + np.array([ -2 , -2 ])
    healthy_type_2 = np.random.randn(number_of_samples, 2) + np.array([ 2 , 2 ])
    healthy = np.vstack([healthy_type_1,healthy_type_2])
    #   concatenation to create the inputs matrix
    inputs_data = np.vstack([healthy,sick])
    #   to every samples created will be associate with an output where 0 is healthy and 1 sick
    
    #   creation of a matrix full of 0
    out_healthy = np.zeros(number_of_samples*2) - 1 # 2x because there is twice more persons
    #   creation of a matrix full of 1
    out_sick = np.zeros(number_of_samples*2) + 1    # 2x because there is twice more persons
    #   concatenated outputs
    outputs_data = np.concatenate((out_healthy, out_sick))
    print (outputs_data)
    #   show datas
    plt.scatter(inputs_data[:, 0], inputs_data[:, 1], s=30, c=outputs_data, cmap=plt.cm.Spectral)
    plt.show()
    return inputs_data, outputs_data

def init_variables():
    """
        Init model variables (weights and bias)
    """
    global weight1
    global weight2

    weights_layer_1 = np.random.normal(size=2)
    weights_layer_2 = np.random.normal(size=2)
    weight1 = np.vstack([weights_layer_1,weights_layer_2])
     
    weight2 = np.random.normal(size=2)
    
    bias_1 = np.zeros((2,1))
    bias_2 = 0

    return bias_1, bias_2

def training (inputs, outputs, bias_1, bias_2):
    global weight1
    global weight2
    
    learning_rate = 0.1
    
    for a in range (40):
        for i in range (400):
    
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
            
        i = 0
        
    print (weight1)
    print (weight2)
    
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
    
    
    inputs,outputs = database()
    bias_1, bias_2 = init_variables()
    training(inputs, outputs, bias_1, bias_2)
    
    my_input=np.array([ -2 , 2 ]) #position of a person
    predict(my_input,bias_1, bias_2)