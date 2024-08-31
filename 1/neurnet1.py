# 1 neuron, 4 inputs

inputs = [3.5, 2.1, 1.8]
weights = [2.2, 4.1, 7.5]
bias = 3

output = 0
for i in inputs:
    output += inputs[i] * weights[i]
output += bias


print(output) 
