# Slusarz_BSc_thesis
The following repository is a tool during a Bachelor thesis titled "Mathematical properties of neural networks trained on artificial datasets" by Natalia Ślusarz which can also be found among the files to provide context for the experiments and can be read to see already achieved results.

<p align="center">
<img src="https://github.com/ndslusarz/Slusarz_BSc_thesis/blob/main/images/spiral_1.png" width="300" height="300" />
</p>

This tool is centered around experiments involving an artificial dataset in the form of a double Archimedean spiral, complexity of which can be controlled by paramters (mathematical definitions of which can be found in the thesis) and multi-layer feedforward neural networks (NNs) trained on this dataset. Its main aim is to provide methods that allow for analysis of various parameters of the NN in order to analyse its learning process.

## Installation

To use this tool simply clone this reporitory and open the code files in your preferred `Python 3.7.0` editor (originally developed in Spyder 3.3.1 through Anaconda Distribution 1.9.12).

Do make sure to have the following libraries installed:
- `matplotlib                3.3.2`
- `scikit-learn              0.23.2`
- `numpy                     1.19.2`
- `pytorch                   1.3.1`


## File structure

Let us now explore the files provided with this tool as well as the provided functionalities.

### Files:
1. `spiral`
2. `nn_class`
3. `nn_class_small`
4. `eigenvalues_functions`
5. `accuracy_functions`
6. `sample_experiments`


### 1. spiral

This file includes the class implementing the dataset - double Archimedean spiral. Its shape can be controlled by providing the radius as well as max_phi (a proxy of entanglement, a maxiumum angle at which the spiral will stop).
It also has a method allowing for adding random noise to the datset for purposes of testing.

### 2. nn_class

This is the file containg the main implementation of the neural network which is a ultilayer, feed-forward NN with user having control over the number and size of layers.
Currently the network has a choice of activation function for inner layers - tanh and ReLU - although more PyTorch functions can be easily added. Softmax was used in the final layer. 
Parameters such as number of epochs, batch size and optimiser can also be specifed. The network takes data in the form of numpy arrays and converst them to PyTorch tensors internally to allow for more flexibility with input data.

This class contains majority of functions involving gathering raw data from the network - its weight matrix for different layers and epochs of training, as well as a way to check whether there exists a vanishing gradient problem.

It provides implementation of FGSM and BIM algorithms for generating adversarial examples for the purposes of any experiments involving adversarial robustness.

### 3. nn_class_small

A modified version of the above. It always has a single hidden layer with two neurons for the purposes of analysis how the points are mapped during training to achieve linearly separable data during the classification.

It provides methods to visualise the mapping during various epochs and with customisable batch size. 

It also provides information of th Euclidean distance between the final position of points and their original one.

It does not however provide methods for generating adversarial examples. 

It has been created as a separate class as for the purposes of point mapping it takes care to keep track of order of the data throught the training process hence has an addutional paramter that could cause unpredictable behaviour for some nn_class functions and was not needed for the purposes of other experiments.

### 4. eigenvalues_functions

This is a collection of methods used for extraction and analysis of data from weight matrices of NN. 

These include extraction of Frobenius norm, eigenvalues, number of complex eigenvalues and norms of eigenvalues for various layers of the NNs from nn_class.

An overreaching function is provided that integrates most of the avilable functionalites, giving a comparison of Frobenius norm as well as eigenvalues for all layers at the end of training. This provides a good starting point that can be modified to facilitate other experiments.

### 5. accuracy_functions

A collection of methods mostly for comparison and visualise the difference in their accuracy depending on complexity of the dataset, size of the network or adversarial example used in testing.

### 6. sample_experiments

This file provides some examples of usage of methods from other files in the form of a few sample experiments on different neural networks.


