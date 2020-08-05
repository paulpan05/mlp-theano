import theano.tensor as T

def sigmoid(Z):
    """Sigmoid activation function.

    Args:
        Z (np.ndarray): the output of network layer before activation.
    """
    return 1 / (1 + T.exp(-Z))

def relu(Z):
    """ReLU activation function.

    Args:
        Z (np.ndarray): the output of network layer before activation.
    """
    return T.maximum(0, Z)

def back_sigmoid(dA, Z):
    """Sigmoid activation function.

    Args:
        dA (np.ndarray): the change in activation of layer.
        Z: The output of layer before activation.
    """
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def back_relu(dA, Z):
    """ReLU activation function.

    Args:
        dA (np.ndarray): the change in activation of layer.
        Z: The output of layer before activation.
    """
    dZ = T.set_subtensor(dA[Z <= 0], 0)
    return dZ
