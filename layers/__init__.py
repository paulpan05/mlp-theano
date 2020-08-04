import numpy as np
import theano
import theano.tensor as T
from layers.activations import sigmoid, relu, back_sigmoid, back_relu

class Dense:
    def __init__(self, units, n_inputs=None, activation='relu'):
        self.units = units
        self.n_inputs = n_inputs
        self.act_name = activation
        if activation == 'relu':
            self.activation = relu
            self.back_activation = back_relu
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.back_activation = back_sigmoid
        else:
            raise Exception('Non-supported activation function')
    def __load(self):
        self.weights = theano.shared(0.10 * np.random.randn(self.units, self.n_inputs), 'weights')
        self.biases = theano.shared(np.zeros((self.units, 1)), 'biases', broadcastable=(False, True))
    def __forward(self, A_prev):
        self.Z = T.dot(self.weights, A_prev) + self.biases
        self.A = self.activation(self.z)
    def __backward(self, dA, A_prev):
        m = A_prev.shape[1]
        dZ = self.back_activation(dA, self.Z)
        self.dW = T.dot(dZ, A_prev.T) / m
        self.db = T.sum(dZ, axis=1, keepdims=True) / m
        self.dA = T.dot(self.weights.T, dZ)
    def __getstate__(self):
        return (self.weights, self.biases, self.units, self.n_inputs, self.act_name)
    def __setstate__(self, state):
        W, b, u, n, act = state
        self.weights = W
        self.biases = b
        self.units = u
        self.n_inputs = n
        self.act_name = act
        if act == 'relu':
            self.activation = relu
            self.back_activation = back_relu
        elif act == 'sigmoid':
            self.activation = sigmoid
            self.back_activation = back_sigmoid
        else:
            raise Exception('Non-supported activation function')