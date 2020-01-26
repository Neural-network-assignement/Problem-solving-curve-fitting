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
#################################################################################################################

def test(var):


    outputs = np.tanh(np.dot(synaptic_weights,var))  # maintenant on va faire la somme des inputs * weight  
    
    return outputs

#################################################################################################################

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

#################################################################################################################

def init_variables():
    
    global synaptic_weights

    synaptic_weights = np.random.random((1, 1)) # on randomise le poid de chaques synapses
    
    

#################################################################################################################
    
def training(inputs,training_outputs,activation_function):

    
    if (activation_function == 0): #NUL
        
        def function(x):
            
            return 0
        
    if (activation_function == 1): # Sigmoid
        
        def function(x):
            
            return 1 /(1 + np.exp(-x))
    
    if (activation_function == 2): # Tangent hyperbolic
       
        def function(x):
            
            return np.tanh(x)
    
    if (activation_function == 3): # Cosinus
        
        def function(x):
            
            return np.cos(x)
   
    if (activation_function == 4): # Gaussian
        
        def function(x):
            
            return np.exp(-(x**2)/2)
     
    
    error_total = 0

    for iteration in range(1):
            
        input_layer = inputs
        
        outputs = function(np.dot(input_layer,synaptic_weights))  # maintenant on va faire la somme des inputs * weight  
    
        error = training_outputs - outputs
        
        for i in range(len(error)):
            
            error_total += error[i]**2
        error_total/len(inputs)
    
    return error_total[0]
    #print (synaptic_weights)



#################################################################################################################
    
def training2(inputs,training_outputs,activation_function):
    

    weights = np.random.random((1, 1))*10 # on randomise le poid de chaques synapses

    
    if (activation_function == 0): #NUL
        
        def function(x):
            
            return 0
        
    if (activation_function == 1): # Sigmoid
        
        def function(x):
            
            return 1 /(1 + np.exp(-x))
    
    if (activation_function == 2): # Tangent hyperbolic
       
        def function(x):
            
            return np.tanh(x)
    
    if (activation_function == 3): # Cosinus
        
        def function(x):
            
            return np.cos(x)
   
    if (activation_function == 4): # Gaussian
        
        def function(x):
            
            return np.exp(-(x**2)/2)
     
    
    error_total = 0


    for iteration in range(1):
            
        input_layer = inputs
        
        outputs = function(np.dot(input_layer,weights))  # maintenant on va faire la somme des inputs * weight  
    
        error = training_outputs - outputs
        
        for i in range(len(error)):
            
            error_total += error[i]**2
        error_total/len(inputs)
    
    return error_total[0], weights[0][0]
    #print (synaptic_weights)


#################################################################################################################
    


def training3(inputs,training_outputs,activation_function,weights):
    


    
    if (activation_function == 0): #NUL
        
        def function(x):
            
            return 0
        
    if (activation_function == 1): # Sigmoid
        
        def function(x):
            
            return 1 /(1 + np.exp(-x))
    
    if (activation_function == 2): # Tangent hyperbolic
       
        def function(x):
            
            return np.tanh(x)
    
    if (activation_function == 3): # Cosinus
        
        def function(x):
            
            return np.cos(x)
   
    if (activation_function == 4): # Gaussian
        
        def function(x):
            
            return np.exp(-(x**2)/2)
     
    
    error_total = 0


    for iteration in range(1):
            
        input_layer = inputs
        
        outputs = function(np.dot(input_layer,weights))  # maintenant on va faire la somme des inputs * weight  
    
        error = training_outputs - outputs
        
        for i in range(len(error)):
            
            error_total += error[i]**2
        error_total/len(inputs)
    
    return error_total[0]
    #print (synaptic_weights)


#################################################################################################################



    
def random_2D_searching():
    
    while(True):
        error = 50
        activation_function = np.random.randint(4, size = 1)
        error = training(inputs,outputs,activation_function) # Null
        if (error < 0.1):
            break
    print ("n'est fini :3")
    
##################################################################################################################
    
def find_activation_function():
    
    error = []
    
    for i in range(4):
        
        error.append(training(inputs,outputs,i))
    
    best = np.min(error)
    for a in range(4):
        if error[a] == best:
            best = a
            
    return best



##################################################################################################################    
    

def init_pop(nb_pop):
    population = np.zeros((nb_pop,10))

    for i in range(nb_pop):
        population[i][4],population[i][3] = training2(inputs,outputs,activation)
        population[i][3] = round(population[i][3],2)
        population[i][4] = round(population[i][4],2)
        population[i][0] = i
        
    mins = population[:,4]
    mins = np.amin(mins)
    
    for i in range(nb_pop):
        if population[i][4] == mins:
            best = i

    for i in range(nb_pop):
        
        population[i][5] = population[best][3]
        population[i][6] = population[best][4]
        seed(random())
        population[i][9] = round(random(),2)*10
        



    #print("   n  x(t-1) y(t-1)  x(t)  y(t)   xg   yg    xr")    
    #print(population)
    
    population_vector = np.zeros((nb_pop,4))


    for i in range(4):
        
        population_vector[i][0] = i

        
        if(population[i][3] - population[i][5] > 0):
            
            population_vector[i][1] = round(1 * random(),2)
            
        if(population[i][3] - population[i][5] < 0):
            
            population_vector[i][1] =round(-1 * random(),2)
        
        if(population[i][3] - population[i][7] > 0):
            
            population_vector[i][2] = round(1 * random(),2)
            
        if(population[i][3] - population[i][7] < 0):
            
             population_vector[i][2] = round(-1 * random(),2) 
    #print("   n       vg      vr")
    #print(population_vector)
    return population,population_vector


#################################################################################☻"
    
def see_function():
    y = []    
    for i in range(-100,100):
        
        y.append(training3(inputs,outputs,activation,i))
    plt.plot(y,"-g")

    plt.show()


######################################################################################    


if __name__ == '__main__':
    
    
    inputs, outputs = get_data()   
    init_variables()
    activation = find_activation_function() # return 3
    nb_pop = 4
    population,population_vector = init_pop(nb_pop)
    print("   n  x(t-1) y(t-1)  x(t)  y(t)   xg   yg   xp     yp    xr")    
    print(population)
    print(" ")

    for i in range(100):
    
          
        #print(population)
        #print("  n    vg    vr")
        #print(population_vector)
        
        
        population[:,1] = population[:,3]# x(t-1) = x(t)
        population[:,2] = population[:,4]# y(t-1) = y(t)
        
        # move forward by adding xg + xr + xp
        # y(t) change according to the new "x"
        for i in range(nb_pop):
            
            direction = population_vector[i][1] + population_vector[i][2] + population_vector[i][3]
            
            population[i][3] = population[i][3] + direction
            population[i][4] = round(training3(inputs,outputs,activation,population[i][3]),2)
            
            
        #find the new minimum
        mins = population[:,4]
        mins = np.amin(mins)
        for i in range(nb_pop):
            if population[i][4] == mins:
                best = i
            
        #change de groupe best position + random new position to go
        for i in range(nb_pop):
            
            population[i][5] = population[best][3]
            population[i][6] = population[best][4]
            
            seed(random())
            population[i][9] = round(random(),2)*10
            
        # change the new best personal solution if so
        for i in range(nb_pop):

            if population[i][2] < population[i][4]:
                
                population[i][7] = population[i][4]
                population[i][8] = population[i][5]
            
                
        for i in range(nb_pop):
            
            population_vector[i][0] = i
    
            
            if(population[i][3] - population[i][5] > 0):
                
                population_vector[i][1] = round(1 * random(),2)
                
            if(population[i][3] - population[i][5] < 0):
                
                population_vector[i][1] =round(-1 * random(),2)
            
            if(population[i][3] - population[i][7] > 0):
                
                population_vector[i][2] = round(1 * random(),2)
                
            if(population[i][3] - population[i][7] < 0):
                
                population_vector[i][2] = round(-1 * random(),2) 

            if(population[i][3] - population[i][9] > 0):
                
                population_vector[i][3] = round(1 * random(),2)
                
            if(population[i][3] - population[i][9] < 0):
                
                population_vector[i][3] = round(-1 * random(),2)
            
    print("   n    x(t-1)   y(t-1)  x(t)   y(t)    xg    yg     xp     yp      xr")  
    print(population)
    print(population_vector)
    see_function()
