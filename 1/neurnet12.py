#using batches in softmax
import numpy as np
import nnfs #book data

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)

#                                                                               axis=0 if rows(vertical), axis=1 if columns(horizontal)
#                                                                               axis=None to sum up the whole matrix

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)
print(np.sum(norm_values))