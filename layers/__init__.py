import numpy as np
import theano
import theano.tensor as T
from layers.activations import sigmoid, relu, back_sigmoid, back_relu

class Dense:
    def __init__(self, units, n_inputs=None, activation='relu'):
        self.units = units
        self.n_inputs = n_inputs
        if activation == 'relu':
            self.activation = relu
            self.back_activation = back_relu
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.back_activation = back_sigmoid
        else:
            raise Exception('Non-supported activation function')
    def load(self, weights=None, biases=None):
        if weights:
            self.weights = weights
        else:
            self.weights = theano.shared(0.10 * np.random.randn(self.n_inputs, self.units), 'weights')
        if biases:
            self.biases = biases
        else:
            self.biases = theano.shared(np.zeros((1, self.units)), 'biases', broadcastable=(True, False))
    def forward(self, inputs):
        self.Z = T.dot(inputs, self.weights) + self.biases
        self.A = self.activation(self.z)
    def backward(self, dA):
        dZ = self.back_activation(dA, self.Z)