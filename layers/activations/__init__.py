import theano.tensor as T

def sigmoid(Z):
    return 1 / (1 + T.exp(-Z))

def relu(Z):
    return T.maximum(0, Z)

def back_sigmoid(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def back_relu(dA, Z):
    dZ = T.set_subtensor(dA[Z <= 0], 0)
    return dZ
