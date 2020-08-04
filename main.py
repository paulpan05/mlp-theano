import numpy as np
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
import theano.tensor as T

X = [[1, 2, 3, 2.5],
[2.0, 5.0, -1.0, 2.0],
[-1.5, 2.7, 3.3, -0.8]]

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = theano.shared(0.10 * np.random.randn(n_inputs, n_neurons), 'weights')
        self.biases = theano.shared(np.zeros((1, n_neurons)), 'biases', broadcastable=(True, False))
    def forward(self, inputs):
        self.output = T.dot(inputs, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        self.output = T.maximum(0, inputs)

layer1 = Dense(4, 5)
layer2 = Dense(5,2)
relu = ReLU()
layer1.forward(X)
layer2.forward(layer1.output)
relu.forward(layer2.output)

print(relu.output.eval())