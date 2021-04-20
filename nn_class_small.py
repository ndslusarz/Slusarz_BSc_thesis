# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 19:54:56 2021

@author: Natalia Ślusarz
"""


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial import distance

'''
A version of nn_class for small network with one hidden layer consisting of 2 neurons (binary classification)
contains all functions for experiments with point mapping 
(how the input data from 2D space gets mapped by a NN in an attempt to achieve linear separability)
'''


SEED = 56424
LEARNING_RATE = 0.01
MOMENTUM = 0.9

class MLP_small(nn.Module):
    
    """hidden layers size: [2] one layer with 2 neurons, [2,5] two layers with 2 and 5 neurone consecuitevley
    hidden_layers - hidden layers size: [2] one layer with 2 neurons, [2,5] two layers with 2 and 5 neurone consecuitevley
    activ - supports 'tanh' and 'relu'
    out_size - size of the output layer

    """
    def __init__(self, out_size, activ='tanh'):
        super(MLP_small, self).__init__()
        hidden_sizes = [2]
        self.hidden=[]
        self.weights={}
        self.biases={}
        self.predictions = {}
        self.activ = activ
        torch.manual_seed(SEED)

        '--------------- input layer'
        self.hidden.append(nn.Linear(2, hidden_sizes[0]))
        self.add_module("input layer", self.hidden[-1])

        '---------------- hidden layers'
        
        for k in range(len(hidden_sizes)-1):
            self.hidden.append(nn.Linear(hidden_sizes[k], hidden_sizes[k+1]))
            self.add_module("hidden_layer"+str(k), self.hidden[-1])
        '---------------- output layer'
        self.out = nn.Linear(hidden_sizes[-1], out_size)



    def forward(self, x):

        for layer in self.hidden:
            if self.activ == 'tanh':
                x = torch.tanh(layer(x))
            if self.activ == 'relu':
                x = torch.relu(layer(x))
            self.all_outputs[self.epoch].append(x.detach().numpy())
        output = self.out(x)
        return output
    
    def load_data(self, x, y, z):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        z = torch.from_numpy(z)
        return TensorDataset(x,y,z) 
    
    ''' 
    learning rate and momentum are global variables
    takes:
        epochs, batch size, optimiser = Adam/SGD
        data in the form of x_train, y_train and ordering (to keep track of any shuffling, needed for later experiments) 
    '''
    def train(self, epochs, batches, x_train, y_train, ordering, optimiser, shuffle_val=True):
        self.all_outputs = {}
        self.all_output_labels = {}
        self.epoch_number = epochs
        criterion = nn.CrossEntropyLoss()
        self.epoch_number = epochs
        if optimiser == 'SGD':
            optimiser = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        elif optimiser == 'Adam':
            optimiser = torch.optim.Adam(self.parameters())
        else:
            raise Exception
        trainset = self.load_data(x_train, y_train, ordering)
        self.losses = []
        
        for epoch in range(epochs):
            self.epoch = epoch
            self.all_outputs[epoch] = []
            self.all_output_labels[epoch] = []
            self.input_data = []
            self.order = []
            data_batches = torch.utils.data.DataLoader(trainset, batch_size=batches, shuffle=shuffle_val)
            
            'input_data needed later for calculating euclidean distance'
            for x_train_bt, y_train_bt, order in data_batches:
                self.input_data.append(x_train_bt.detach().numpy())
                self.order.append(order.detach().numpy())
                
            for x_train_bt, y_train_bt, order in data_batches:
                self.all_output_labels[epoch].append(y_train_bt.detach().numpy())
                
                optimiser.zero_grad()
                outputs = self(x_train_bt.float())
                y_train_bt = y_train_bt.long().flatten()
                loss = criterion(outputs, y_train_bt)
                loss.backward()
                optimiser.step()

            self.losses.append(loss.item())
            self.get_weights(epoch)


            
            
    def test(self, x_test, y_test, ordering, batches=1):
        self.test_accuracy = 0
        self.test_loss = 0.0
        testset = self.load_data(x_test, y_test, ordering)
        data_batches = torch.utils.data.DataLoader(testset, batch_size=batches, shuffle=True)
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        criterion = nn.CrossEntropyLoss()
        


        for data, target, order in data_batches:
           
            output = self(data.float())
            target = target.long().flatten()
            loss = criterion(output, target)
            self.test_loss += loss.item()*data.size(0)
            pred = torch.Tensor(np.argmax(output.detach().numpy(), axis=1))

            correct = np.squeeze(pred.eq(target.data.view_as(pred)))

            for i in range(batches):
                
                label = target.data[i].int()
                class_correct[label] += correct.item()
                class_total[label] += 1
          
        self.test_loss = self.test_loss/len(data_batches.dataset)
        print('Test Loss: {:.6f}\n'.format(self.test_loss))
        
        for i in range(len(class_total)):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    str(i), 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
        self.test_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
        
    ''''
    gets weights from a single epoch, every layer
    does not return the matrix - fills an appropriate class variable
    '''
    def get_weights(self, epoch):
        self.weights[epoch] = []
        for layer in self.hidden:
            self.weights[epoch].append(layer.weight.detach().numpy())
        self.weights[epoch].append(self.out.weight.detach().numpy())
       
        
       
    '''
    returns matrix with values of a weight matrix for specific epoch and layer
    '''
    def get_weights_layer(self, epoch, layer):
        weights = self.weights.get(epoch-1)[layer-1]
        return weights
    

    '''
    visualise weight matrix as a heatmap in specified layer and epoch of training
    '''
    def visualise_weight_matrix(self, epoch, layer):
        fig, ax = plt.subplots()
    
        matrix = self.weights.get(epoch-1)[layer-1]
        plt.xlabel("Neurons")
        plt.ylabel("Inputs")
        plt.title('Weight matrix for epoch '+str(epoch)+ ' and layer '+str(layer))
        (n, m) = matrix.shape
            
    
        for i in range(n):
            for j in range(m):
                el = matrix[i,j]
                ax.text(j, i, str(round(el,2)), va='center', ha='center')
        
        plt.imshow(matrix, cmap='coolwarm')
        
        
    '''
    returns eigenvalues of weight matrix in specified layer and epoch
    '''
    def get_eigenvalues(self, epoch, layer):
        weights = self.weights.get(epoch-1)[layer-1]
        matrix = np.linalg.eigvals(weights)
        return matrix
    
    

    '''
    plot where the 2D input data is mapped to after given epoch
    '''
    def plot_mapping(self, epoch):
        plt.rc('axes', titlesize=15)
        data = []
        labels = []
        for index in range(len(self.all_outputs[epoch])):
            output = self.all_outputs[epoch][index]
            label = self.all_output_labels[epoch][index]
            if data == []:
                data = output
                labels = label
            else:
                data = np.vstack((data, output))
                labels = np.vstack((labels, label))

        plt.figure(dpi=300)
        plt.scatter(data[:,0], data[:,1], c=labels, cmap='RdBu')
        plt.title('After epoch '+str(epoch+1))
        plt.show()    

    '''
    get and plot euclidean distance between original data point and the one it was mapped to
    plots graphs for whole dataset and seperate ones for both clasess
    '''
    def plot_distance(self, size):
        out_data = []
        in_data = []
        orders = []
        labels = []
        for index in range(len(self.all_output_labels[self.epoch_number-1])):
            output = self.all_outputs[self.epoch_number-1][index]
            label = self.all_output_labels[self.epoch_number-1][index]
            inp = self.input_data[index]
            order = self.order[index]
            if out_data == []:
                out_data = output
                in_data = inp
                orders = order
                labels = label
            else:
                out_data = np.vstack((out_data, output))
                in_data = np.vstack((in_data, inp))  
                orders = np.hstack((orders, order)) 
                labels = np.vstack((labels, label))
                
            
                
        distances = [-1 for x in range(size)] 
        dist_from_origin = [-1 for x in range(size)] 
        for i in range(len(orders)):
            index = orders[i]
            distances[index] = distance.euclidean(out_data[i], in_data[i])
            dist_from_origin[index] = distance.euclidean(in_data[i], 0.)

        new_dist = []
        new_dist_origin = []
        for el in distances:
            if el!= -1:
                new_dist.append(el)
        for el in dist_from_origin:
            if el!= -1:
                new_dist_origin.append(el)

        final_x = []
        final_y = []
        final_label = []
        for i in range(0, len(new_dist), 20):
            final_x.append(new_dist_origin[i])
            final_y.append(new_dist[i])
            final_label.append(labels[i])
            
        x0 = []
        x1 = []
        y0 = []
        y1 = []
        
        for i in range(len(final_x)):
            if final_label[i]==0:
                x0.append(final_x[i])
                y0.append(final_y[i])
            if final_label[i]==1:
                x1.append(final_x[i])
                y1.append(final_y[i])
        
        plt.figure(dpi=800)
        plt.rc('font', size=10) 

        plt.bar(x0, y0, width=0.2, alpha=0.7)
        plt.bar(x1, y1, width=0.2, alpha = 0.7)
        plt.title('Euclidean distance between the original datapoints\n and their final mapping (ReLu)')
        plt.xlabel('Original distance from origin')
        plt.ylabel('Distance between input value and mapping')
        plt.show()
        plt.figure(dpi=800)
        plt.rc('font', size=10) 
        plt.bar(x1, y1, width=0.2, alpha = 0.7)
        plt.title('Spiral 1')
        plt.xlabel('Original distance from origin')
        plt.ylabel('Distance between input value and mapping')
        plt.show()  
        plt.figure(dpi=800)
        plt.rc('font', size=10) 
        plt.bar(x0, y0, width=0.2, alpha = 0.7)
        plt.title('Spiral 2')
        plt.xlabel('Original distance from origin')
        plt.ylabel('Distance between input value and mapping')
        plt.show() 
        
    
    '''
    plots decision boundary for a given test set (x - data, y - labels)
    '''
    def plot_decision_boundary(self,x,y):
        X_test_t = torch.FloatTensor(x)
        #y_hat_test = self(X_test_t)
        y_hat_test_class = np.argmax(self(X_test_t).detach().numpy(), axis=1)
        print(len(y_hat_test_class))
        #y_hat_test_class = np.where(y_hat_test.detach().numpy()<0.5, 0, 1)
        test_accuracy = np.sum(y.flatten()==y_hat_test_class) / len(y)
        print('Accuracy: ', test_accuracy)
        
        x_min, x_max = x[:, 0].min()-0.5, x[:, 0].max()+0.5
        y_min, y_max = x[:, 1].min()-0.5, x[:, 1].max()+0.5
        spacing = min(x_max - x_min, y_max - y_min) / 100
        XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing), np.arange(y_min, y_max, spacing))
        data = np.hstack((XX.ravel().reshape(-1,1), YY.ravel().reshape(-1,1)))
        data_t = torch.FloatTensor(data)
    

        db_prob = np.argmax(self(data_t).detach().numpy(), axis=1)

        Z = db_prob.reshape(XX.shape)
  

        plt.figure(figsize=(12,8))
        plt.contourf(XX, YY, Z, cmap=plt.cm.Accent, alpha=0.5)
        plt.scatter(x[:,0], x[:,1], c=y, cmap=plt.cm.Accent)
        plt.show()

       
