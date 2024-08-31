#softmax activation function
#it's more compatible with back propagation

import math

layer_outputs = [4.8, 1.21, 2.385]

E = math.e

exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)
    
print(exp_values)

#normalization function (so everything will add up to approximately 1)
norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print(norm_values)
print(sum(norm_values))