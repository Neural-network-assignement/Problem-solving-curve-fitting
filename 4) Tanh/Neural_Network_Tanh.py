import numpy as np





def derivative_function(x):
    
    return (1 - (x*x))




def get_data():
    
    global width

    data=np.genfromtxt("1in_tanh.txt",delimiter="")
    shape = np.shape(data)
    
    width = shape[0]
    
    inputs = np.zeros((width,1))
    outputs = np.zeros((width,1))
    
    for i in range(width):
        inputs[i] = data[i][0]
        outputs[i] = data[i][1]
        
    return width,inputs,outputs




def init_variables():

    global weight1
    global weight2

    weight1 = np.random.random((1,3))
    weight2 = np.random.random((3,1)) 
    bias_1 = np.zeros((2,1))
    bias_2 = 0
    


    return bias_1, bias_2






def training (inputs, outputs, bias_1, bias_2,width):
    global weight1
    global weight2
    
    lr = 0.1
    
    hActivation = np.zeros((3, 1), dtype=float)
    oActivation = np.zeros((1, 1), dtype=float)
     
    # outputs of neurons (after sigmoid function)
    iOutput = np.zeros((1, 1), dtype=float)      # +1 for bias
    hOutput = np.zeros((3, 1), dtype=float)  # +1 for bias
    oOutput = np.zeros((1), dtype=float)
     
    # deltas for hidden and output layer
    hDelta = np.zeros((3), dtype=float)
    oDelta = np.zeros((1), dtype=float)   


    #i = np.array([[1],[2]])
    #w = np.array([[2,2],[2,2]]) 
    #o = np.dot(w,i)
    #print(inputs[0][0])
    #print(inputs[0][1])

    for i in range(2000):
        for a in range(width):
            
            
            
            iOutput = inputs[a][0]

         
            #print(iOutput)
            # hidden layer
            
            hActivation = np.dot(iOutput,weight1)
            hOutput = np.tanh(hActivation)
            #print(hOutput)
            
 
            # output layer
            oActivation = np.dot(hOutput,weight2)
            oOutput = np.tanh(oActivation)
            print(oOutput)
            
    
            error = oOutput - np.array(outputs[a], dtype=float) 
             
            # deltas of output neurons
            oDelta = (1 - np.tanh(oActivation)) * np.tanh(oActivation) * error
                     
            # deltas of hidden neurons
            hDelta = (1 - np.tanh(hActivation)) * np.tanh(hActivation) * np.dot(oDelta,weight2.transpose())
            #print(self.hDelta)        
            # apply weight changes
            weight1 += lr * np.dot(iOutput.transpose(),hDelta) 
            weight2 += lr * np.dot(hOutput.transpose(),oDelta)
        i=0
                    

    
def predict ():
    
    
    iOutput = -1

 
    hActivation = np.dot(iOutput,weight1)
    hOutput = np.tanh(hActivation)
    #print(hOutput)
    
 
    # output layer
    oActivation = np.dot(hOutput,weight2)
    oOutput = np.tanh(oActivation)
    print(oOutput)
            
    
if __name__ == '__main__':
    
    
    width,inputs,outputs = get_data()
    bias1,bias2 = init_variables()
    training(inputs,outputs,bias1,bias2,width)
    #print(weight1)
    #sha = weight1.shape
    #print(sha)
    #print(weight2)
    predict()