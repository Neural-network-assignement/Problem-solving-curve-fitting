# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:24:52 2019

@author: Cata Daria
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:19:56 2019

@author: Tim hey not only :p don't forget me!
"""

"""
init_pop adapted

the update have to be updated
the weight have to be retrieved and put into 2 matrix (i and h one vertical the other horizontal)
"""

# import necessary Python packages
import os
import numpy as np
from random import *
import matplotlib as mpl
import matplotlib.pyplot as plt


#################################################################################################################
#################################################################################################################

def get_data():
    
    
    data=np.genfromtxt("1in_sine.txt",delimiter="")
    
    shape = np.shape(data)

    width = shape[0]
    
    inputs = np.zeros((width,1))
    training_outputs = np.zeros((width,1))
    
    for i in range(width):
        
        inputs[i]           = data[i][0]
        training_outputs[i] = data[i][1]
        
    return inputs,training_outputs

#################################################################################################################
def init_variables(nb_neurons):
    
    global synaptic_weights
    weight_ih_train = np.random.uniform(low=-3, high=3, size=(1,nb_neurons))# matrix wight input hidden layer
    weight_ho_train = np.random.uniform(low=-3, high=3, size=(nb_neurons,1))# matrix
    
    return weight_ih_train,weight_ho_train

#################################################################################################################
def find_activation_function(inputs, outputs,weight_ih_train,weight_ho_train):
    
    """
    this function is used at the beginning of the program and it basically set the weight at a specific value
    and then calculate the error for every activation function.
    at the end it sees which one made the least error and return it.
    """
    best = []
    error_func = []
    samples = 10
    for i in range(1,5): # goind through every activation function
        error = 0
        for a in range(samples):
            weight_ih_train = np.random.uniform(low=-1, high=1, size=(1,nb_neurons))# matrix wight input hidden layer
            weight_ho_train = np.random.uniform(low=-1, high=1, size=(nb_neurons,1))# matrix
            error += training(weight_ih_train,weight_ho_train,inputs,outputs,i) # stock it in an array
            
        error = error / samples
        
        error_func.append(error) # find the min value
    #print(error_func)
    best = np.argmin(error_func)+1
    #print(best+1)
    if best == 1:
        print("activation function : sigmoid")
    if best == 2:
        print("activation function : Tanh")
    if best == 3:
        print("activation function : cosinus")
    if best == 4:
        print("activation function : Gaussian")
    
    return 2#best # return the best number associate with the activation function



##################################################################################################################        
def training(weight_ih_train,weight_ho_train,inputs,training_outputs,activation_function):
   
    """
    traning where the weight are already set. this function is made to find the most suitable activation function
    we use it only to find the right activaiton function
    """
    #initialise the weights

    
    shape = np.shape(inputs)
    width = shape[0]
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
     
    
    #init the matrix of the neurons that will do the math
    
    #hidden activation part
    hActivation = np.zeros((nb_neurons, 1), dtype=float)
    #output activation part
    oActivation = np.zeros((1, nb_neurons), dtype=float)
     
    # outputs of neurons (after sigmoid function)
    # for the input
    iOutput = np.zeros((1, 1), dtype=float)      
    # for the hidden layer
    hOutput = np.zeros((nb_neurons, 1), dtype=float) 
    # for the output
    oOutput = np.zeros((1), dtype=float)
     

    error = 0 # init the total mean error
    in_outputs = []
    # go throught all the value in the array
    
    for a in range(width):
        
        iOutput = inputs[a][0] #input
        

        # hidden layer
        
        hActivation = np.dot(iOutput,weight_ih_train)  # hidden layer activation
        hOutput = function(hActivation)        # hiddent layer output
        #print(hOutput)
        
 
        # output layer
        
        oActivation = np.dot(hOutput,weight_ho_train) # output layer activation
        oOutput = function(oActivation)        # output layer output
        #print(oOutput)
        in_outputs.append([iOutput,oOutput])

        error += (np.array(training_outputs[a], dtype=float)-oOutput)**2 # error(t) = error(t-1) + error(t)²
        
    error = error/width # error / number of samples
    
    return float(error) # return the error
    #print (synaptic_weights)



#################################################################################################################
#################################################################################☻"

def see_function(inputs,outputs,activation): 

    plt.plot(outputs,"r")
    #plt.show()

#################################################################################################################


def function_chosen(inputs, weightih, weightho,activation_function):
    """once the function has been chosen and the best weights estimated this is the final function to predict"""
   
     
    shape = np.shape(inputs)
    width = shape[0]
    
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
     
    
    #init the matrix of the neurons that will do the math
    
    #hidden activation part
    hActivation = np.zeros((nb_neurons, 1), dtype=float)
    #output activation part
    oActivation = np.zeros((1, nb_neurons), dtype=float)
     
    # outputs of neurons (after sigmoid function)
    # for the input
    iOutput = np.zeros((1, 1), dtype=float)      
    # for the hidden layer
    hOutput = np.zeros((nb_neurons, 1), dtype=float) 
    # for the output
    oOutput = np.zeros((1), dtype=float)
     
    in_outputs = []
    # go throught all the value in the array
    for a in range(width):
        
        iOutput = inputs[a] #input

        # hidden layer
        
        hActivation = np.dot(iOutput,weightih)  # hidden layer activation
        hOutput = function(hActivation)        # hiddent layer output
        #print(hOutput)
        
 
        # output layer
        
        oActivation = np.dot(hOutput,weightho) # output layer activation
        oOutput = function(oActivation)        # output layer output
        #print(oOutput)
        output=float(oOutput)
        #print(oOutput)
        in_outputs.append([float(iOutput),round(output,4)])
        

        
        
  
    return in_outputs

    
    
    
##################################################################################################################

class swarm_particle :
    def __init__(self, nb_pop, nb_neurons,nb_layer):  #this works for 1 hidden layer if more = more weights it will have to be adapted
        self.nb_layer=nb_layer+1
        nb_weight= nb_neurons*self.nb_layer
        #self.id_neuron=np.zeros((nb_neurons*nb_pop*self.nb_layer, 2))#this one is only there to better understand what's happening
        self.w_position=np.zeros((nb_pop,nb_weight))#nb of popu (lines) weights in networks (1 weight 1 column)
        self.net_error=np.zeros((nb_pop,2)) #last error of each net (id+error)
        self.w_velocity = np.zeros((nb_pop,nb_weight)) #velocity of each particle (1 velocity for 1 network)
        self.w_prev_position= np.zeros((nb_pop,nb_weight)) 
        self.w_best_perso_position= np.zeros((nb_pop,nb_weight)) #keep in memory the best weigts associated to best_group error in each network
        self.best_perso_position_net_error=np.zeros((nb_pop,2)) #error of perso best (id + error)
        self.net_best_group=np.zeros((1,2)) #best network (id + error)
        self.best_position_group=np.zeros((1,nb_weight))#weights of best particle
        self.particle_all=[]        #stock all of the info of each particle
        self.nb_neurons=nb_neurons #number of neurons
        self.nb_pop=nb_pop #number of population for each neuron
        #swarm_particle.weight_selection_NNA()
        self.check_plateau=np.zeros((nb_pop,1))
        
        
        self.w_update_velocity=np.zeros((nb_pop,nb_weight))
        
        
    def init_pop(self, inputs, outputs, activation):         #FAUX FAUX FAUX FAUX FAUX
            
        """
        this function is used to initialize the value of the population. it takes as input the number of the population that you want to create
        then it sotck every parameters in an array.
        """
        
        for net_id in range(self.nb_pop):
            weight_ih_train = np.random.uniform(low=-2, high=2, size=(1,self.nb_neurons))# matrix wight input hidden layer
            weight_ho_train = np.random.uniform(low=-2, high=2, size=(self.nb_neurons,1))# matrix
            #print(weight_ho_train)
            for n in range(self.nb_neurons): #only works if nb_layer=1 else have to adapt
                self.w_position[net_id][n]= weight_ih_train[0][n]
                self.w_position[net_id][n+self.nb_neurons]=weight_ho_train[n][0]
                
            
                
            self.net_error[net_id][0]=int(net_id)  
            self.net_error[net_id][1]=training(weight_ih_train,weight_ho_train,inputs,outputs,activation)#error of the network
            self.w_velocity[net_id][:] =round(np.random.uniform(-1,1),3) #we created velocities with random number between -1 and 1 
        self.best_perso_position_net_error[:]=self.net_error[:]
        self.w_prev_position[:]=self.w_position[:]
        self.w_best_perso_position[:]= self.w_position[:]
        
        #search best particle and keep it as best of the group
        best= np.argmin(self.net_error[:,1])#mins.index(mini)
        self.net_best_group[:]=self.net_error[best][:]
        self.best_position_group[:]=self.w_position[best][:]
         
        
##################################################################################################################    

    def update_velocity(self):
        np.random.seed()
        
        w = 0.5#0.729 # inertia weight
        c1 = 2.9 #cognitive/local weight
        c2 = 2 # social/global weight
        #r1, r2  # cognitive and social randomizations
        r1= round(np.random.uniform(0,1),3)
        r2= round(np.random.uniform(0,1),3)
        # velocity depends on prec velocity, best position of patricle, and best position of any particle
        # move forward by adding:  best group pos-personnal pos + best personnal pos-personnal pos + present velocity
        
        for net_id in range(nb_pop):
            #self.w_velocity[net_id,:] =round(np.random.uniform(-2,2),3) #we REcreated velocities with random numberbetween -1 and 1 
           
            for id in range(self.nb_neurons*self.nb_layer): #nb_neuron*nb_layer=nb_weight
                
                vel_cognitive_social=(c1 * r1 * (self.w_best_perso_position[net_id][id] - self.w_position[net_id][id])) + (c2 * r2 * (self.best_position_group[0][id] - self.w_position[net_id][id]))
                self.w_update_velocity[net_id][id] = (w * self.w_velocity[net_id][id]) +vel_cognitive_social
        #print(str(self.w_update_velocity))
        

##################################################################################################################    
            
    def update_position(self, inputs, outputs, activation,min_boundary, max_boundary):
       weight_ih=np.zeros(shape=(1,self.nb_neurons))
       weight_ho=np.zeros(shape=(self.nb_neurons,1))
                                         
        
       self.w_prev_position[:]=self.w_position[:] #we store the position in the previous position matrix, to keep it for check purpose 
       self.w_position[:] = self.w_position[:]+self.w_update_velocity[:] #we update position with his posirion + his velocity  
        
       #print("new pos")
       #print(self.w_position)
              #for each member of the population
       for net_id in range (self.nb_pop):

           #if the position doesn't change enought then we check if it doesn't move for too long, if it does we reset the particle
           if np.sum(abs(self.w_prev_position[net_id][:] - self.w_position[net_id][:])) < 1*self.nb_neurons*self.nb_layer :
               """print("abs(self.w_prev_position[net_id][:]-self.w_position[net_id][:])")
               print(net_id)
               print(abs(self.w_prev_position[net_id][:]-self.w_position[net_id][:]))"""
               self.check_plateau[net_id][0]+=1
              
           else:
               self.check_plateau[net_id][0]=0

           for i in range(0,self.nb_neurons*self.nb_layer): #only works for 1 lyer  
               if self.check_plateau[net_id][0] > 10 :
                   
                   #print("HEEEEEEEREEEEEEEEEEE")
                   self.w_position[net_id][i]=round(np.random.uniform(-1,1),6)
                   self.w_velocity[net_id][0]=round(np.random.uniform(-1,1),6)
                   
               if self.w_position[net_id][i] > max_boundary :
                   #if the position is over the boundary, the particle respawn somewhere random
                   self.w_position[net_id][i] = round(np.random.uniform(-1,1),6)
                    
               if self.w_position[net_id][i] < min_boundary: #the min and max boundary for the particles, to avoid them going too far
                   
                   self.w_position[net_id][i]=round(np.random.uniform(-1,1),6)
               
               if i < self.nb_neurons: #we put half the weight in a matrix input hidden
                   
                    weight_ih[0][i] = self.w_position[net_id][i]
               else:                 #we put half the weight hidden output
                   
                   weight_ho[i-self.nb_neurons][0] = self.w_position[net_id][i]  
                    #NB;: this function can't accept [:]
               
               
           self.net_error[net_id][1]=training(weight_ih,weight_ho,inputs,outputs,activation)   # new error assoaciate
           
           #check if the particle is stuck at the same optima for too long If yes it reset the whole particle in a new position, velocity.
           if self.check_plateau[net_id][0]>10 :
              self.best_perso_position_net_error[:]=self.net_error[:]
              self.w_prev_position[:]=self.w_position[:]
              self.w_best_perso_position[:]= self.w_position[:]
           
            #check if the error of this particle is the best out of all its history
           if self.net_error[net_id][1] < self.best_perso_position_net_error[net_id][1]:  #if this position is best than his personnal best position, we change it
                self.w_best_perso_position[net_id,:]= self.w_position[net_id,:]
                self.best_perso_position_net_error[net_id][1]=self.net_error[net_id][1]
           
            #check if the best error of this particle is the best of the whole population
           if self.best_perso_position_net_error[net_id][1] < self.net_best_group[0][1]:
               self.net_best_group[:]=self.net_error[net_id][:]
               self.best_position_group[:]=self.w_position[net_id][:]
        
               
#################################################################################################################
######################################################################################            
          
    def weight_selection_NNA(self):
   
        weight_ih=np.zeros(shape=(1,self.nb_neurons))
        weight_ho=np.zeros(shape=(self.nb_neurons,1))
        
        if nb_layer>1:
            print("\n This PSO is not optmized for more than one hidden layer, please wait another month !!!! \n")
        for i in range(self.nb_neurons):

            weight_ih[0][i] = self.best_position_group[0][i]
            weight_ho[i][0] = self.best_position_group[0][i+self.nb_neurons]
            
        return weight_ih,weight_ho

######################################################################################             
    def predict(self, my_input, weightih, weightho,activation):
        in_outputs=[]
            #print("input")
            #print(input_layer)
    
        in_outputs = function_chosen(my_input, weightih, weightho,activation) # maintenant on va faire la somme des inputs * weight  
        #print(in_outputs)
        
        return in_outputs


######################################################################################    
    def __repr__(self): #to get what's inside the object, change them into strings, just to display the infos
        return "<swarm_particle: particles positions = \n" + str(self.position)+ "; \n velocities = \n" + str(self.velocity) + "\n best perso_position of particle = \n " + str(self.best_pers_position) + "\n previous pos= \n" + str(self.prev_position) + ">"


######################################################################################   
    def main(self, tolerated_error,epocmax,min_boundary, max_boundary):
            
        inputs, outputs = get_data()  

        weight_ih_train,weight_ho_train=init_variables(nb_neurons)
        activation = find_activation_function(inputs,outputs,weight_ih_train,weight_ho_train) # return the best activation function
        particle.init_pop(inputs, outputs,activation) # initalize the population
        #self.particle_all=np.stack([self.id_neuron[:],self.position[:],self.prev_position[:],self.velocity[:],self.best_pers_position[:]]) #here we have all the info of all particles 
        """
        #representation=particle.__repr__() #this function is basically to print for an object
        #print(representation) #e print all our population
        print("position of each weight by network")
        print(str(self.w_position[:]))
        print("net error")
        print(str(self.net_error))
        print("velocity")
        print(str(self.w_velocity))
        print("w_best perso")
        print(str(self.w_best_perso_position))
        print("error of the best position")
        print(str(self.best_perso_position_net_error))
        print("prev position")
        print(str(self.w_prev_position))
        """
        print("best error of the pop")
        print(str(self.net_best_group))
        print("best weight of group")
        print(str(self.best_position_group))
        
        
        #self.best_pers_position_net_error=np.zeros((nb_pop,2)) #error of perso best (id + error)
        #self.net_best_group
        
        i=0
        for i in range(0,epocmax):# we train each particles a "epocmax" number of time  
        #while (self.net_best_group[0][1]>tolerated_error):
            particle.update_velocity()
            particle.update_position(inputs, outputs,activation,min_boundary, max_boundary)
            i+=1
            if (self.net_best_group[0][1]<=tolerated_error): #if the error is under the tolerated thrshold then stop
                break
            #print("new pos")
            #print(self.w_position)
        #representation=particle.__repr__() #this function is basically a print for an object
        #print(representation)
        
        see_function(inputs, outputs,activation)
        print("\n Number of Neurons: ", self.nb_neurons)
        print("\n Population Number : ", self.nb_pop)
        print("\n Error Tolerated :", tolerated_error)
        print("\n number of epoch:", i)
        print("\n particle selected:",int(self.net_best_group[0][0]), "  Actual Error :",float(self.net_best_group[0][1]))
        print("\n best weight of the selected particle: ", str(self.best_position_group))
        
        weightih, weightho= particle.weight_selection_NNA()
        #to test my selected weight
        """
        my_input=[-1,-0.3,1,0.5,0,0.8]
        
        output=self.predict(my_input, weightih, weightho,activation)
        
        print("result:")
        print(output)
        """
        new = []
        new_outputs = self.predict(inputs, weightih, weightho,activation) 
        for i in range(len(inputs)):
            new.append(new_outputs[i][1])
        plt.plot(new,"b")
        #plt.plot(output[1,:],"r")
        #plt.show()

if __name__ == '__main__':
                        # define the number of population
        nb_pop =50
        epocmax=300
        tolerated_error=0.002
        nb_neurons=15
        nb_layer=1
        
        min_boundary=-10
        max_boundary=10
        #the min and max boundary for the particles, to avoid them going too far

        particle=swarm_particle(nb_pop,nb_neurons,nb_layer)
        particle.main(tolerated_error,epocmax,min_boundary, max_boundary)
    
