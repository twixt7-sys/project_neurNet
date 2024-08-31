import numpy as np

def sigmoid(z):
    return 1.0/(1.0/1.0 + np.exp(-z))   #this converts any input into a number between zero and 1

print(sigmoid(4.5 * 0.22 + 3.0))    #the input in the form "weight * activation + bias" is used