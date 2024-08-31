import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] + 
         math.log(softmax_output[2]) * target_output[2])

#the higher the confidence the lower the loss where the loss should come closer to zero as the softmax output of the activated one-hot index comes closer to 1
print(loss)