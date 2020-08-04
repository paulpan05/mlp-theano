import numpy as np
import theano
import theano.tensor as T
from layers import Dense
from layers.activations import back_relu

X = [[1, 2, 3, 2.5],
[2.0, 5.0, -1.0, 2.0],
[-1.5, 2.7, 3.3, -0.8]]

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

# layer1 = Dense(4, 5)
# layer2 = Dense(5,2)
# layer1.forward(X)
# layer2.forward(layer1.output)

# print(layer2.output.eval())

weights = theano.shared(0.10 * np.random.randn(4, 5), 'weights')
biases = theano.shared(np.zeros((1, 5)), 'biases', broadcastable=(True, False))
Z = T.dot(X, weights) + biases