import numpy as np
import theano
import theano.tensor as T
from layers.activations import sigmoid, relu, back_sigmoid, back_relu

class Dense:
    """Dense layer of a neural network.

    Attributes:
        units (int): Number of units in layer.
        n_inputs (int): Number of inputs in layer.
        act_name (str): The name of the activation function.
        activation (function): The function reference to the activation function.
        back_activation (function): The function reference to the backprop activation function.
        weights (theano.compile.sharedvalue.SharedVariable): The weights of the layer.
        biases (theano.compile.sharedvalue.SharedVariable): The biases of the layer.
        dW (np.ndarray): The change in weights of the layer.
        db (np.ndarray): The change in biases of the network.
    """

    def __init__(self, units, n_inputs=None, activation='relu'):
        """Initializes the dense layer.
        
        Args:
            units (int): Number of units in layer.
            n_inputs (int, optional): Number of inputs in layer.
            activation (str, optional): The name of the activation function.
        """
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
        """Loads the randomly-generated weights of the network."""
        self.weights = theano.shared(0.10 * np.random.randn(self.units, self.n_inputs), 'weights')
        self.biases = theano.shared(np.zeros((self.units, 1)), 'biases', broadcastable=(False, True))

    def __forward(self, A_prev):
        """ Forward propagate the layer.

        Args:
            A_prev: The activation of the previous layer.
        """
        self.Z = T.dot(self.weights, A_prev) + self.biases
        self.A = self.activation(self.Z)

    def __backward(self, dA, A_prev):
        """Back propagate the layer.

        Args:
            dA: The change in activation of the current layer.
            A_prev: the activation of the previous layer.
        
        Returns:
            np.ndarray: The gradient of the next layer.
        """
        m = A_prev.shape[1]

        # Backward activation function
        dZ = self.back_activation(dA, self.Z)

        # Calculate the change in weights and biases
        self.dW = T.dot(dZ, A_prev.T) / m
        self.db = T.sum(dZ, axis=1, keepdims=True) / m

        # Return the gradient of the next layer
        return T.dot(self.weights.T, dZ)

    def __getstate__(self):
        """Gets the state of the object to be pickled by Python.

        Returns:
            (np.ndarray, np.ndarray, int, int, string): The state to put into a pickle file.
        """
        return (self.weights, self.biases, self.units, self.n_inputs, self.act_name)

    def __setstate__(self, state):
        """Sets the state of the object from pickled information.

        Args:
            state (np.ndarray, np.ndarray, int, int, string): The state of the pickled object.
        """
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