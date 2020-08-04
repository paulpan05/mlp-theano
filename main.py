import numpy as np
import theano
import theano.tensor as T
from layers import Dense
from models import Sequential
from mnist import MNIST

X = np.array([[1, 0.8, 1, 0],
[1, 1, 0.9, 0],
[0.8, 1, 0.8, 1]])

Y = np.array([[0, 1, 0, 0],
[1, 0, 1, 0],
[0, 0, 0, 1]], dtype=theano.config.floatX)

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

mndata = MNIST('./samples')
images, labels = mndata.load_training()
print(len(labels))
print(len(images))

# model = Sequential()
#model.add(Dense(512, n_inputs=3))
#model.add(Dense(3, activation='sigmoid'))
#model.compile()
#model.fit(X, Y, 100)
