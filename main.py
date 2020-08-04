import numpy as np
import theano
import theano.tensor as T
from layers import Dense
from models import Sequential

X = np.array([[1, 2, 3, 2.5],
[2.0, 5.0, -1.0, 2.0],
[-1.5, 2.7, 3.3, -0.8]])

Y = np.array([[0, 1, 0, 0],
[1, 0, 1, 0],
[0, 0, 0, 1]], dtype=theano.config.floatX)

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

model = Sequential()
model.add(Dense(512, n_inputs=3))
model.add(Dense(3, activation='sigmoid'))
model.compile()
model.fit(X, Y,1)