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


min_boundary=-10
max_boundary=10 #the min and max boundary for the particles, to avoid them going too far
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

    synaptic_weights = np.random.uniform(low=-3, high=3, size=(1,1)) # on randomise le poid de chaques synapses

#################################################################################################################
    
def training(inputs,training_outputs,activation_function):
   
    """
    traning where the weight are already set. this function is made to find the most suitable activation function
    we use it only to find the right activaiton function
    """
    
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
            
        input_layer = inputs   # input
        
        outputs = function(np.dot(input_layer,synaptic_weights))  # maintenant on va faire la somme des inputs * weight : ex : sigmoid(weight * inputs) 
    
        error = training_outputs - outputs # error for every value in matrix form
        
        for i in range(len(error)):
            
            error_total += error[i]**2   # this is the mean square value
        error_total=error_total/len(inputs)
        
        
    return error_total[0]
    #print (synaptic_weights)



#################################################################################################################
    
def training2(inputs,training_outputs,activation_function):

    """
    for this function we randomize the weights every time that we call this function.
    it is doing the exact same thing as the 1st one but we also return the weight
    this is used for initialize the population of the swarm. we need a x and y axis
    
    x = weight
    y = error
    
    so we need to return both of them to get the position.

    """

    #weights = np.random.random((1, 1))*100 # on randomise le poid de chaques synapses
    weights = np.random.uniform(-2.0,2.0, size=(1,1))

    
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
        error_total=error_total/len(inputs)
    
    return weights[0][0], error_total[0]
    #print (synaptic_weights)


#################################################################################################################
    


def training3(inputs,training_outputs,activation_function,weights):
    
    """ this one is for the moving part. the particles moves and then we need to find to new current location in y which is the error in our case.
        so this function return the new error assosiate with the new weight.
        which is exactly as finding f(x) for a specific x.
    
    
    """
    
   
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
        error_total=error_total/len(inputs)
    
    return error_total[0]
    #print (synaptic_weights)

                 
#################################################################################☻"

def see_function(inputs,outputs,activation): 
    """y = []    
    for i in range(-1000,3000):
        a = 0.001*i
        
        y.append(training3(inputs,outputs,activation,a))"""
    plt.plot(outputs,"-g")

    #plt.show()

#################################################################################################################


def function_chosen(inputs,weight,activation_function):
    """once the function has been chosen and the best weights estimated this is the final function to predict"""
   
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
     
    outputs = function(np.dot(inputs,weight))  # maintenant on va faire la somme des inputs * weight  
    outputs=float(outputs)
    return outputs

####################################################################################
    
def random_2D_searching():
    
    """
    this function randomize a the weight and calcul the error associate.
    while this error is > 0.9 it keeps calculating.
    
    this function has one big flaw : it doesn't keep in memory where it's been before. so the program can stay in places that we already know
    """
    while(True):
        error = 50
        activation_function = np.random.randint(4, size = 1)
        error = training2(inputs,outputs,activation_function) # Null
        if (error < 0.9):
            break
    print ("n'est fini :3")
    
##################################################################################################################
    
def find_activation_function(inputs, outputs):
    
    """
    this function is used at the beginning of the program and it basically set the weight at a specific value
    and then calculate the error for every activation function.
    at the end it sees which one made the least error and return it.
    """
    
    error = []
    
    for i in range(1,5): # goind through every activation function
        
        error.append(training(inputs,outputs,i)) # stock it in an array
    
    best = np.argmin(error)+1 # find the min value
    
    if best == 1:
        print("activation function : sigmoid")
    if best == 2:
        print("activation function : Tanh")
    if best == 3:
        print("activation function : cosinus")
    if best == 4:
        print("activation function : Gaussian")
            
    return best # return the best number associate with the activation function




##################################################################################################################    

class swarm_particle :
    def __init__(self, nb_pop):
        
        self.position=np.zeros((nb_pop, 2))
        self.velocity = np.zeros((nb_pop,2))
        self.prev_position= np.zeros((nb_pop,2))
        self.best_pers_position= np.zeros((nb_pop,2))
        self.best_group=np.zeros((1,2)) 
        self.particle_all=[]        #stock all of the info of each particle
        self.update_velocities=np.zeros((nb_pop,2))
        self.nb_pop=nb_pop


    def init_pop(self,nb_pop, inputs, outputs, activation):
        #particle=[]
    
        """
        this function is used to initialize the value of the population. it takes as input the number of the population that you want to create
        then it sotck every parameters in an array.
        """
        
        for i in range(nb_pop):
            weight,error=training2(inputs,outputs,activation)

            self.position[i][0]=round(weight,3)
            self.position[i][1]=round(error,5)
            self.velocity[i][0] =round(np.random.uniform(-1,1),3) #we created velocities with random number between -1 and 1
            self.velocity[i][1] =round(np.random.uniform(-1,1),3) #we created velocities with random number between -1 and 1
            
            
        self.best_pers_position[:]= self.position[:] # as of now, the best and prev are the present position
        self.prev_position[:]=self.position[:]
        
            
        #Now we search the actual best among the randomly created positions   
        mins = self.position[:,1] # we stock the error in a matrix
        mins = np.min(mins)   # we find the min value of it
        for i in range(nb_pop):
            #we search for which number of particles has the min value
            if self.position[i][1] == mins:
                best = i        
        # this loop stock the new best group weight and the error associate in the matrix
        # this is basiacally the best position in the group        
        
        self.best_group[:] = self.position[best,:] # Best weight group
     
##################################################################################################################    

    def update_velocity(self):
        np.random.seed()
        #self.velocity[:,0] =round(np.random.uniform(-1,1),3) #we created velocities with random number between -1 and 1
        #self.velocity[:,1] =round(np.random.uniform(-1,1),3) #we created velocities with random number between -1 and 1
        w = 0.4 # inertia weight
        c1 = 1 #cognitive/local weight
        c2 = 1.5 # social/global weight
        #r1, r2  # cognitive and social randomizations
        r1= round(np.random.uniform(0,1),3)*10
        r2= round(np.random.uniform(0,1),3)*10
        # velocity depends on prec velocity, best position of patricle, and best position of any particle
        # move forward by adding:  best group pos-personnal pos + best personnal pos-personnal pos + present velocity

        vel_cognitive_social=(c1 * r1 * (self.best_pers_position[:] - self.position[:])) + (c2 * r2 * (self.best_group[0] - self.position[:]))
        self.update_velocities[:] = (w * self.velocity[:]) +vel_cognitive_social


##################################################################################################################    
            
    def update_position(self, nb_pop, inputs, outputs, activation):
        global min_boundary
        global max_boundary #the min and max boundary for the particles, to avoid them going too far
        
        for i in range(nb_pop):
            
            self.prev_position[i]=self.position[i] #we store the 
            #print("check pos i before")
            #print(self.position[i])
            self.position[i][0]=self.position[i][0]+self.update_velocities[i][0] #we update position with his posirion + his velocity 
            #print("check posi after")
            #print(self.position[i])
            self.position[i][1]=round(training3(inputs,outputs,activation,self.position[i][0]),5)   # new error assoaciate
                        #NB;: this function can't accept [:]
            if self.position[i][0]>max_boundary : #if the position is over the boundary, the particle respawn somewhere random
                weight,error=training2(inputs,outputs,activation)
                self.position[i][0]=round(weight,5)
                self.position[i][1]=round(error,5)
                
            if self.position[i][0]<min_boundary:
                weight,error=training2(inputs,outputs,activation)
                self.position[i][0]=round(weight,5)
                self.position[i][1]=round(error,5)
                
            if self.position[i][1] < self.best_group[0][1]:  #if this position is best than the previous team best, we change it
                self.best_group[:] = self.position[i,:] # Best weight group
                # this is basiacally the best position in the group  
                
            if self.position[i][1] < self.best_pers_position[i][1]:  #if this position is best than his personnal best position, we change it
                self.best_pers_position[i,:]= self.position[i,:]
            

######################################################################################             
    def predict(self, my_input, weight,activation):
        in_outputs=[]
        for i in range(len(my_input)):
            input_layer = my_input[i]
            #print("input")
            #print(input_layer)
    
            output = function_chosen(input_layer,weight,activation) # maintenant on va faire la somme des inputs * weight  
            in_outputs.append([input_layer,output])
        
        return in_outputs


######################################################################################    
    def __repr__(self): #to get what's inside the object, change them into strings, just to display the infos
        return "<__main__.swarm_particle: particles positions = \n" + str(self.position)+ "; \n velocities = \n" + str(self.velocity) + "\n best perso_position of particle = \n " + str(self.best_pers_position) + "\n previous pos= \n" + str(self.prev_position) + ">"


######################################################################################   
    def main(self, tolerated_error,epocmax):
            
        inputs, outputs = get_data()   
        init_variables()
        activation = find_activation_function(inputs,outputs) # return the best activation function
         
        particle.init_pop(nb_pop, inputs, outputs,activation) # initalize the population
        self.particle_all=np.stack([self.position[:],self.prev_position[:],self.velocity[:],self.best_pers_position[:]]) #here we have all the info of all particles 
        
        representation=particle.__repr__() #this function is basically to print for an object
        #print(representation) #e print all our population
        
        print ("first best particle [weight error]")
        print(str(self.best_group))
        for i in range(0,epocmax):# we train each particles a "epocmax" number of time

            if any(self.position[:,1]<= tolerated_error): #if the error is under the tolerated thrshold then stop
                break
            particle.update_velocity()
            particle.update_position(nb_pop, inputs, outputs,activation)
    
        representation=particle.__repr__() #this function is basically a print for an object
        #print(representation)
        
        see_function(inputs, outputs,activation)

        print("\n Population Number : ", self.nb_pop)
        print("\n Error Tolerated :", tolerated_error)
        print("\n number of epoch:",i+1)
        print("\n Weight particle selected:",float(self.best_group[0][0]), "  Actual Error :",float(self.best_group[0][1]))
       
        
        #to test my selected weight
        my_input=[1,0.5,0,0.8]
        output1=self.predict(my_input, self.best_group[0][0],activation)
        output=self.predict(inputs, self.best_group[0][0],activation)
        
        print("result:")
        print(output1)
        
        plt.plot(output,'r')
        
        plt.show()
        return i+1

        

if __name__ == '__main__':
                        # define the number of population
    nb_pop = 25
    epocmax=500
    tolerated_error=0.001
    particle=swarm_particle(nb_pop)
    particle.main(tolerated_error,epocmax)
    
    
    """
    epoch_final = []
    samples = 10
    for a in range(1,50):
        np_pop = a
        epoch = 0
        particle=swarm_particle(nb_pop)
        for i in range(samples):
            epoch += particle.main(tolerated_error,epocmax)**2
        epoch = epoch / samples
        epoch_final.append(epoch)
    
    plt.plot(epoch_final,"-b")

    plt.show()
    """
    
