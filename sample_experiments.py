# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:01:23 2021

@author: Natalia Ślusarz

Some ready experiments providing example of use of some methods
"""


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt

from spiral import *
from nn_class import *
from accuracy_functions import *
from nn_class_small import *
from eigenvalues_functions import *


# Set global parameters:
optimiser = 'Adam'
hidden_layers = [50,50,50,50,50] 
epochs = 100 
batchsize = 50  


#generate spiral data
spiral = Data_spiral(ent=8, min_phi = 1, num_samples=1000, radius=2)
spiral.generate_data()
x_train, x_val, y_train, y_val = train_test_split(spiral.x_data, spiral.y_data, test_size=0.33, random_state=42)



#initialise a nn (ch)
net = MLP(hidden_layers, 2, activ='tanh' )
#train the nn
net.train(epochs,batchsize, x_train, y_train, optimiser)
#plot decision boundary for test set
net.plot_decision_boundary(x_val, y_val)



#use the overreaching function from eigenvalues_functions
#it will return comparison of Frobenius norm, eigenvalues etc for nns trained on spirals with entanglement in range(min_val, max_val, skip)
max_val =12
min_val = 2
skip = 2
eigen_all_layers(max_val, min_val, skip, hidden_layers, epochs, batchsize, 'relu')



#test the networks against adversarial examples
print('Test against adversarial examples - BIM')
epsilon = 0.5
iterations = 4
net.generate_adversarial_bim(x_val, y_val, epsilon, iterations)



#test the network against adversarial examples with varying epsilons and visualise results
#tests networks with tanh and relu agaianst each other
epsilons = [0.1, 0.25, 0.5, 1, 1.5, 2]
alpha = 0.3
iterations = 4
BIM_accuracy(epsilons, hidden_layers, epochs, batchsize, alpha, iterations, x_train, y_train, x_val, y_val)



#analysing the point mapping using nn_class_small
#this needs apart from data consisting of points and labels a third array that tracks position of points during training
spiral = Data_spiral(ent=5, min_phi = 1, num_samples=1000, radius=2)
spiral.generate_data()
ordering = np.arange(0, 2000, 1)
x_train, x_val, y_train, y_val, ord_train, ord_val = train_test_split(spiral.x_data, spiral.y_data, ordering, test_size=0.33, random_state=55)

net = MLP_small(2, activ='tanh')
batchsize=50
net.train(epochs, batchsize, x_train, y_train, ord_train, 'Adam')
net.plot_mapping(epochs-1)



