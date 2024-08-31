import random

#Mini-batch Stochastic Gradient Descent Function
#It means taking a only a sample of inputs from a batch and approximating its gradient

def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    #Training data --> A list of tuples '(x, y)' representing the training inputs and the desired outputs
    #Evaluation will occur after each epoch(batch) if test data is provided
    # eta -> learning rate
    if test_data:
        n_test = len(test_data)
        n = len(training_data)
    for j in range(epochs):
        random.shuffle(training_data)
        mini_batches = [test_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch_size)
    if test_data:
        print("Epoch {0}: {1} / {2}").format(j, self.evaluate(test_data), n_test)
    else:
        print("Epoch {0} complete").format(j)