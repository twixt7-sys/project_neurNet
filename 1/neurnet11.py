#coding the softmax activation function using numpy
import math
import numpy as np
import nnfs #book data

nnfs.init()

layer_outputs = [4.8, 1.21, 2.385]

E = math.e

exp_values = np.exp(layer_outputs)

norm_values = exp_values / np.sum(exp_values)

print(norm_values)
print(sum(norm_values))

'''
softmax activation function:

input --> exponentialize (y = e^x) --> normalize (e^1 / (e^1 + e^2 + ... + e^n)) --> Output as a part of 1
'''