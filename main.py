import pickle
import numpy as np
import theano
from mnist import MNIST
from layers import Dense
from models import Sequential

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

def parse_data():
    mndata = MNIST('./samples')
    images, labels = mndata.load_training()

    vocab = set()
    for label in labels:
        vocab.add(label)
    vocab = sorted(vocab)
    Y = []
    for label in labels:
        one_hot = [0] * len(vocab)
        one_hot[label] = 1
        Y.append(one_hot)
    X = np.array(images).T / 255
    Y = np.array(Y).T
    return (X, Y)

X, Y = parse_data()

model = Sequential()
model.add(Dense(1024, n_inputs=X.shape[0]))
model.add(Dense(1024))
model.add(Dense(1024))
model.add(Dense(Y.shape[0], activation='sigmoid'))
model.compile()

# model = pickle.load(open('model.p', 'rb'))

model.fit(X, Y, 1, learning_rate=0.003)
