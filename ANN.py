import numpy as np

class Network(object):
    
    def __init__(self, neurons):

        self.Neurons = neurons
        self.Layers = len(neurons)
        # Initialize the weights as a matrix of random weights. 
        self.Weights = [np.random.rand(x,y) for x, y in zip(neurons[:-1], neurons[1:])]
        # Initialize the bias as a vector of random biases
        self.Biases = [np.random.rand(1,x) for x in neurons[1:]]

    def activationfunction(self, currentActivation):
        return 1.0/(1.0+np.exp(-currentActivation))

    def activationfunctionderivitive(self, currentActivation):
        return self.activationfunction(currentActivation)*(1-self.activationfunction(currentActivation))

    def feed_forward(self, X):
        a = X
        activations = [a]
        inputs = [X]

        for b, w in zip(self.Biases, self.Weights):
            z = a.dot(w) + b
            inputs.append(z)
            a = self.activationfunction(z)
            activations.append(a)
        return [activations,inputs]

    def Output(self, input):
        return self.feed_forward(input)[0][-1]

    def TrainWithBackPropagation(self, input, expectedOutput, numIters):

        for j in range(numIters):
            # Feedforward
            # Get the activation of the first layer, which is the input. 
            res = self.feed_forward(input)
            activations = res[0]
            inputs = res[1]

            # Backprop
            # Set yHat error and yHat delta. 
            a_error = expectedOutput - activations[-1]
            errors = [a_error]
            a_Delta = a_error * self.activationfunctionderivitive(inputs[-1])
            activationDeltas = [a_Delta]
            # Update the weights of the last layer. 
            self.Weights[-1] += inputs[-1].T.dot(a_Delta)
            weightDeltas = [self.Weights[-1]]

            # For each layer, backpropagate through the network and adjust the weights. 
            for l in xrange(2, self.Layers):
                # Compute the error and delta of the next layer
                a_error = a_Delta.dot(self.Weights[-l + 1].T)
                errors.append(a_error)
                a_Delta = a_error * self.activationfunctionderivitive(inputs[-l])
                activationDeltas.append(a_Delta)
                # Adjust the layers weights. 
                self.Weights[-l] += inputs[-l - 1].T.dot(a_Delta)
                weightDeltas.append(self.Weights[-l])

        # Return the results of the change. 
        return activations[-1]
