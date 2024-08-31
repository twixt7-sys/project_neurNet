#vectorizing function
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def feedforward(self, a):
    """Return the output of the network if "a" is input"""
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a) + b)
    return a