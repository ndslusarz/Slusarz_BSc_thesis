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
from torch.utils.data import TensorDataset
from matplotlib import colors

SEED = 56424
LEARNING_RATE = 0.01
MOMENTUM = 0.9


class MLP(nn.Module):
    
    """
    hidden_layers - hidden layers size: [2] one layer with 2 neurons, [2,5] two layers with 2 and 5 neurone consecuitevley
    activ - supports 'tanh' and 'relu'
    out_size - size of the output layer
    """
    def __init__(self, hidden_sizes, out_size, activ='tanh'):
        super(MLP, self).__init__()
        self.hidden=[]
        self.weights={}
        self.biases={}
        self.activ = activ
        self.predictions = {}
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
        output = self.out(x)
        return output
    
    def load_data(self, x, y):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return TensorDataset(x,y) 

    
    ''' 
    learning rate and momentum are global variables
    takes:
        epochs, batch size, data, optimiser = Adam/SGD
    '''
    def train(self, epochs, batches, x_train, y_train, optimiser):
        criterion = nn.CrossEntropyLoss()
        self.epoch_number = epochs
        
        if optimiser == 'SGD':
            optimiser = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        elif optimiser == 'Adam':
            optimiser = torch.optim.Adam(self.parameters())
        else:
            raise Exception
            
        trainset = self.load_data(x_train, y_train)
        self.losses = []
        for epoch in range(epochs):
            self.epoch = epoch
            data_batches = torch.utils.data.DataLoader(trainset, batch_size=batches, shuffle=True)
            for x_train_bt, y_train_bt in data_batches:

                optimiser.zero_grad()
                outputs = self(x_train_bt.float())
                y_train_bt = y_train_bt.long().flatten()
                loss = criterion(outputs, y_train_bt)
                loss.backward()
                optimiser.step()
            self.losses.append(loss.item())
            self.get_weights(epoch)
            self.predictions[epoch] = self(torch.from_numpy(x_train).float())
            
            
    
    def test(self, x_test, y_test, batches=10):
        self.test_accuracy = 0
        self.test_loss = 0.0
        testset = self.load_data(x_test, y_test)
        data_batches = torch.utils.data.DataLoader(testset, batch_size=batches, shuffle=True)
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        criterion = nn.CrossEntropyLoss()

        for data, target in data_batches:
            
            output = self(data.float())
            target = target.long().flatten()
            loss = criterion(output, target)
            self.test_loss += loss.item()*data.size(0)
            pred = torch.Tensor(np.argmax(output.detach().numpy(), axis=1))
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            for i in range(batches):
                
                label = target.data[i].int()
                class_correct[label] += correct[i].item()
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
    
    ''''
    visualise weight matrix as a heatmap in specified layer and epoch of training
    '''
    def visualise_weight_matrix(self, epoch, layer):
        fig = plt.figure(figsize=(6,6), dpi=300)
        ax = fig.add_subplot(111)
        plt.xlabel("Neuron in layer "+str(layer))
        plt.ylabel("Input from layer "+str(layer-1))
        plt.title('Weight matrix for epoch '+str(epoch)+ ' and layer '+str(layer))
        
        matrix = self.weights.get(epoch-1)[layer-1]
        (n, m) = matrix.shape
        #cmap for printing values only without the colormap but as a matplotlib image
        #cmap = colors.ListedColormap(['white'])
        ax.grid(False)
        # Minor ticks
        ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 10, 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', linewidth=2)
        im = ax.imshow(matrix, cmap='Spectral')
        for i in range(n):
             for j in range(m):
                 el = matrix[i,j]
                 ax.text(j, i, str(round(el,2)), va='center', ha='center')
                 
        fig.colorbar(im)
        plt.show()

    '''
    returns eigenvalues of weight matrix in specified layer and epoch
    '''
    def get_eigenvalues(self, epoch, layer):
        weights = self.weights.get(epoch-1)[layer-1]
        matrix = np.linalg.eigvals(weights)
        return matrix
    
    def hook(self, module, inp, out):
        self.outputs.append(out)
        
        
    '''
    returns the amount of gradients = 0 for each layer after training 
    used to check if there exists vanishing gradient problem
    '''
    def gradients_get(self):
        counter = {}
        for layer in self.hidden:
            grads = layer.weight.grad
            counter[layer] = 0
            for el in grads.flatten():
                if el == 0.:
                    counter[layer] += 1
        layer = self.out
        grads = layer.weight.grad
        counter[layer] = 0
        for el in grads.flatten():
            if el == 0.:
                counter[layer] += 1
        print ('Number of gradients = 0 ', counter)
        return counter
    
      
    '''
    plots decision boundary for a given test set (x - data, y - labels)
    '''
    def plot_decision_boundary(self,x,y):
        X_test_t = torch.FloatTensor(x)
        y_hat_test_class = np.argmax(self(X_test_t).detach().numpy(), axis=1)
        print(len(y_hat_test_class))
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
        
        
    
    '''
    generate adversarial examples - FGSM method
    and use them as a test set
    returns how much of adversarial examples were still classified correctly (accuracy as a fraction)
    '''
    def generate_adversarial_fgsm(self, point_x, point_y, epsilon):
        criterion = nn.CrossEntropyLoss()
        
        testset = self.load_data(point_x, point_y)
        data_batches = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
        correct = 0
        all_count = 0
        for data, target in data_batches:
            all_count += 1
            data.requires_grad = True
            output = self(data.float())
            target = target.long().flatten()
            loss = criterion(output, target)
            loss.backward(retain_graph=True) 
        
            x_grad = torch.sign(data.grad)
            x_adv = data + epsilon * x_grad
            output_adv = self(x_adv.float())
            pred = torch.Tensor(np.argmax(output_adv.detach().numpy(), axis=1))
            if target[0].float() == pred[0]:
                correct += 1
        print ('Got ', correct, ' out of ', all_count) 
        return correct/all_count
         
    
    '''
    generate adversarial examples - BIM method
    and use them as a test set
    returns how much of adversarial examples were still classified correctly (accuracy as a fraction)
    '''
    def generate_adversarial_bim(self, point_x, point_y, epsilon, iterations):
        criterion = nn.CrossEntropyLoss()
        alpha = 0.5*epsilon
        testset = self.load_data(point_x, point_y)
        data_batches = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
        all_count = 0
        correct = 0
        for data, target in data_batches:

            
            all_count += 1
            rand = np.random.uniform(-epsilon, epsilon)
            data[0][0]+=rand
            rand = np.random.uniform(-epsilon, epsilon)
            data[0][1]+=rand
            data.requires_grad = True
            x_adv = data
            x_adv.requires_grad = True
            for i in range(iterations):

                output = self(data.float())
                target = target.long().flatten()
                loss = criterion(output, target)
                loss.backward(retain_graph=True) 
            
                x_grad = torch.sign(data.grad)
                x_adv = data + alpha * x_grad
                
                #make sure it lies in the epsilon ball
                if (x_adv[0][0]<data[0][0]-epsilon):
                    x_adv[0][0] = data[0][0]-epsilon
                if (x_adv[0][0]>data[0][0]+epsilon):
                    x_adv[0][0] = data[0][0]+epsilon
                if (x_adv[0][1]<data[0][1]-epsilon):
                    x_adv[0][1] = data[0][1]-epsilon
                if (x_adv[0][1]>data[0][1]+epsilon):
                    x_adv[0][1] = data[0][1]+epsilon
                output_adv = self(x_adv.float())
                pred = torch.Tensor(np.argmax(output_adv.detach().numpy(), axis=1))    
                data = x_adv
                data = data.detach()
                data.requires_grad = True


            if target[0].float() == pred[0]:
                correct += 1
        print ('Got ', correct, ' out of ', all_count)
        return correct/all_count

