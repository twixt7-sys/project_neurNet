#making a 4-input, 2 layered 5-neuron network with 2 outputs using python classes and np.random as weights

import numpy as np

np.random.seed(0)

X =    [[1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
#                  inputs        outputs
    def __init__(self, n_inputs, n_neurons):                                    #python constructor (makes the neurons)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases                #python method (passes the data from 1 neuron to another)
    
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)