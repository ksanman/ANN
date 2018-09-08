import ANN as ann
import numpy as np

network = ann.Network([2,5,1])
input = np.array([[0,0],[1,0],[0,1],[1,1]])
output = np.array([[0],[0],[0],[1]])
trained = network.TrainWithBackPropagation(input, output, 5000)
res = network.Output(input)
print res