# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np

def sigmoid (x):                #on crée la fonction sigmoid de la prof
    return 1 /( 1 + np.exp(-x))

def sigmoid_deriv (x):
    return x * (1 - x)

training_inputs = np.array([[0,0,1],        #on definie sous forme de matrice les inputs qui sont au nombre de 3
                           [1,1,1],        # tu verras avec le training output que seul les inputs ayant un 1 dans la 1ere colonnes
                           [1,0,1],         # on des outputs à 1
                           [0,1,1]])
    
training_outputs = np.array([[0,1,1,0]]).T # on definie les output des inputs cela va peremttre d'entrainer notre algo avec des 
                                            # exemple. le "T" veux dire transposer. en gros la matrice horizontale deviens verticale
np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1 # on randomise le poid de chaques synapses

print ('random starting synaptic weight')           # et on le print
print (synaptic_weights)

for iteration in range (20000):
        
    input_layer = training_inputs
    
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))  # maintenant on va faire la somme des inputs * weight  

    error = training_outputs - outputs
    
    adjustment = error * sigmoid_deriv(outputs)
    
    synaptic_weights += np.dot(input_layer.T, adjustment)
    
print (synaptic_weights)
print ('outputs after training')
print (outputs)
    