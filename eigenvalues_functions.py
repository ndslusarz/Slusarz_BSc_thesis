# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 14:04:18 2021

@author: Natalia Ślusarz


a file with methods involving weight matrix, eigenvalues and Frobenius norm using CED and nn_class

"""
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd


from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt




from spiral import *
from nn_class import *


'''
plots eigenvalues, their norm, Frobenius norm for weight matrices in all layers (not only final)
for entanglement in range(min_val, max_val, skip)

this function comprises of most of the singular functionalities offered in functions below
'''
def eigen_all_layers(max_val, min_val, skip, hidden_layers, epochs, batchsize, activ):
    data = {}
    frobenius = {}
    norms = {}
    for i in range(min_val, max_val, skip):
        spiral = Data_spiral(ent=i, min_phi = 2, num_samples=1000, radius=2)
        spiral.generate_data()
        x_train, x_val, y_train, y_val = train_test_split(spiral.x_data, spiral.y_data, test_size=0.33, random_state=42)
        net = MLP(hidden_layers,2, activ)
        net.train(epochs, batchsize, x_train, y_train, 'Adam')
        data[i] = {}
        frobenius[i] = {}
        norms[i] = {}
        for k in range(1, len(hidden_layers)):
            eigen = net.get_eigenvalues(epochs, k+1)
            weights = net.get_weights_layer(epochs, k+1)
            data[i][k] = eigen
            frobenius[i][k] = np.linalg.norm(weights)
            norms[i][k] = []
            for el in eigen:
                norms[i][k].append(abs(el))
        #nwt.test(x_val, y_val)
        net.plot_decision_boundary(x_val, y_val)
    counter = count_complex(data, hidden_layers)
    plot_frobenius_all_layers(min_val, max_val, skip, len(hidden_layers), frobenius)
    plot_norms_all_layers(min_val, max_val, skip, len(hidden_layers), norms)
    plot_how_many_complex(counter)
        


'''
returns norms of given data 
data - dictionary with elements being arrays of eigenvalues
'''        
def get_eigen_abs(data):
    norms = {}
    for key, val in data.items():
        norm = []
        for el in val:
            norm.append(abs(el))
        norms[key] = norm
    return norms

'''
returns Frobenius norm of matrices
data - dictionary with elements being matrices (2D arrays)
'''  
def get_Frobenius_norms(data):
    frobenius = {}
    for key, val in data.items():
        frobenius[key] = np.linalg.norm(val)
    return frobenius


'''
plot norms of eigenvalues for networks trained on datasets with increased phi (entanglement)
takes a dictionary of norms with keys being phi and elements arrays of norms
'''
def plot_eigen_abs(data): 
    x_data = []
    y_data = []
    plt.xlabel("Maximum phi - entanglement")
    plt.ylabel("Value")
    plt.title("Norm of eigenvalues of the final weight matrix of NN trained on spirals with increasing entanglement")
    for key, val in data.items():
        for el in val:
            x_data.append(key)
            y_data.append(el)
    plt.scatter(x_data, y_data)
    plt.show()
  
'''
Plots values of eigenvalues of final weight matrix of NN trained on spirals with increasing entanglement
takes dictionary with keys being entanglement and elements arrays of values

Eigenvalues are plotted on a plane in case of presence of complex numbers
'''
def plot_eigen(data): 

    x_data = []
    y_data = []
    labels = []
    
    for key, val in data.items():
        for el in val:
            x_data.append(el.real)
            y_data.append(el.imag)
            labels.append(key)
    df = pd.DataFrame(dict(x=x_data, y=y_data, label=labels))

    groups = df.groupby('label')

    fig, ax = plt.subplots()
    ax.margins(0.05) 
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
    ax.legend()
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title("Eigenvalues of final weight matrix of NN trained on spirals with increasing entanglement")
    plt.show()
    
'''

Plots Frobenius norm of final weight matrix of NN trained on spirals with increasing entanglement
takes dictionary with keys being entanglement and elements arrays of values
'''
    
def plot_frobenius(data):
    x_data = []
    y_data = []
    plt.xlabel("Maximum phi - entanglement")
    plt.ylabel("Value")
    plt.title("Frobenius norm of eigenvalues of final weight matrix of NN trained on spirals with increasing entanglement")
    for key, val in data.items():
            x_data.append(key)
            y_data.append(val)
    plt.scatter(x_data, y_data)
    plt.show()
    
    
'''
plots eigenvalues, their norm, Frobenius norm of final weight matrix
for entanglement in range(min_val, max_val, skip)
'''
def eigen_final_layer( max_val, min_val, skip, hidden_layers, epochs, batchsize):
    data = {}
    for i in range(min_val, max_val, skip):
        spiral = Data_spiral(ent=i, min_phi = 2, num_samples=1000, radius=2)
        spiral.generate_data()
        x_train, x_val, y_train, y_val = train_test_split(spiral.x_data, spiral.y_data, test_size=0.33, random_state=42)
        net = MLP(hidden_layers,1)
        net.train(epochs, batchsize, x_train, y_train)
        data[i] = net.get_eigenvalues(epochs, len(net.hidden)-1)
        net.plot_decision_boundary(x_val, y_val)
    plot_eigen( data)
    data_abs = get_eigen_abs(data)
    plot_eigen_abs(data_abs)
    frobenius = get_Frobenius_norms(data)
    plot_frobenius(frobenius)
    
'''
plots  Frobenius norm for weight matrices in all layers (not only final)
for entanglement in range(min_val, max_val, skip)
'''
def plot_frobenius_all_layers(min_val, max_val, skip, y, data):
    plt.rc('font', size=10)
    fig, ax = plt.subplots(figsize=(8,6))

        
    ax.set_title('Frobenius norm of weight matrix with increasing entanglement', size=18) # Title
    ax.set_ylabel('Frobenius norm', fontsize = 15) # Y label
    ax.set_xlabel('Entanglement of the training set', fontsize = 15) # X label  
    x_data = []
    y_data = []       
    for i in range(1, y):
        for k in range(min_val, max_val, skip):
            y_data.append(data[k][i])
            x_data.append(k)
        name = 'Layer '+str(i)
        ax.plot(x_data, y_data, linestyle='--', marker='o', label = name)
        x_data.clear()
        y_data.clear()
    ax.legend(fontsize=15)
     
    plt.show()
    
'''
plots  max-min value of norm of eigenvalues for weight matrices in all layers (not only final)
for entanglement in range(min_val, max_val, skip)
'''
def plot_norms_all_layers(min_val, max_val, skip, y, data):

    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle("Max - min eigenvalue of the weight matrix with increasing entanglement", size=18)

    
    ax.set_ylabel('Max-min', fontsize = 15) # Y label
    ax.set_xlabel('Entanglement of the training set', fontsize = 15)    
    x_data = []
    y_data = []       
    for i in range(1, y):
        for k in range(min_val, max_val, skip):

            y_data.append(np.max(data[k][i])-np.min(data[k][i]))
            x_data.append(k)
        name = 'Layer '+str(i)

        ax.plot(x_data, y_data, linestyle='--', marker='o', label = name)
        x_data.clear()
        y_data.clear()
    ax.legend(fontsize=15)
    plt.show()
    
'''
count number of complex eigenvalues of the final weight matrix
'''
def count_complex(data, hidden_layers):
    counter = {}
    for key, val in data.items():
        eigenvals = val[len(hidden_layers)-1]
        counter[key] = 0
        for eig in eigenvals:
            if eig.imag!=0:
                counter[key]+=1
    return counter

'''
plot number of complex eigenvalues for networks with varying entanglement
takes dictionary that is returned from count_complex
'''
def plot_how_many_complex(data):
    x_data = []
    y_data = []
    for key, val in data.items():
        x_data.append(key)
        y_data.append(val)
    plt.xlabel("Entanglement of the training set")
    plt.ylabel("")
    plt.title("Amount of complex eigenvalues of the final weight matrix\n of NNs trained on spiral datasets with increasing entanglement")
          
    plt.scatter(x_data, y_data)
    plt.show()
 

