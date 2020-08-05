import pickle
import theano
import theano.tensor as T

class Sequential:
    """The sequential model class for a neural network.
    
    Attributes:
        layers (list): The layers of the neural network.
        best_loss (int): The best loss value of the training.
    """
    def __init__(self):
        """Initializes the empty list and the best loss."""
        self.layers = []
        self.best_loss = None

    def add(self, layer):
        """Adds a layer to the neural network.

        Args:
            layer (Dense): The layer to add to the network.
        """
        if len(self.layers) == 0:
            if not layer.n_inputs:
                raise Exception('Need to have n_inputs for layer.')
        else:
            layer.n_inputs = self.layers[-1].units
        self.layers.append(layer)

    def compile(self):
        """Compiles the network with the given inputs and outputs.

        Note:
            Don't compile if loading from file.
        """
        for layer in self.layers:
            layer._Dense__load()

    def fit(self, X, Y, epochs, batch_size=32, learning_rate=0.01):
        """Fit the neural network model.

        Args:
            X (np.ndarray): The input to the network.
            Y (np.ndarray): The outputs of the network.
            epochs (int): The number of epochs to train.
            batch_size (int, optional): The number of samples in a training batch.
            learning_rate (float, optional): The learning rate of gradient descent.
        """
        n = 0
        n_samples = X.shape[1]
        for i in range(epochs):
            while n * batch_size < n_samples:
                X_cur = X[:, n * batch_size : min((n + 1) * batch_size, n_samples - 1)]
                Y_cur = Y[:, n * batch_size : min((n + 1) * batch_size, n_samples - 1)]
                Y_hat = self.__feedforward(X_cur)
                cost = self.__get_cost_value(Y_hat, Y_cur)
                total_cost = T.sum(cost).eval()
                if self.best_loss == None or total_cost < self.best_loss:
                    self.best_loss = total_cost
                accuracy = self.__get_accuracy_value(Y_hat, Y_cur).eval()
                print('Epoch '+ str(i + 1) +' - ' + str(n + 1) + '/' + str((n_samples // batch_size) + 1) + ' Loss: ' + str(total_cost) + ' Accuracy: ' + str(accuracy), end='\r', flush=True)
                self.__backprop(X_cur, Y_hat, Y_cur)
                self.__update(learning_rate)
                pickle.dump(self, open('model.p', 'wb'))
                n += 1

    def __feedforward(self, X):
        """Feedforward propagate step of training.

        Args:
            X (np.ndarray): The batch of input to train.
        Returns:
            np.ndarray: The observed output of the network.
        """
        A = X
        for layer in self.layers:
            layer._Dense__forward(A)
            A = layer.A
        return A

    def __backprop(self, X, Y_hat, Y):
        """Backpropagation step of training.

        Args:
            X (np.ndarray): The batch of input to train.
            Y_hat (np.ndarray): The output of the network during the feedforward step.
            Y (np.ndarray): The correct outputs.
        """
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape.eval())

        # Compute the gradient of the output layer
        dA = - ((Y / Y_hat) - ((1 - Y) / (1 - Y_hat)))
        i = len(self.layers) - 1
        while i >= 0:

            # Get activation of the previous layer
            A_prev = None
            if i == 0:
                A_prev = X
            else:
                A_prev = self.layers[i-1].A
            
            # Compute gradient for the next layer down
            dA = self.layers[i]._Dense__backward(dA, A_prev)
            i -= 1

    def __update(self, learning_rate):
        """Updates the weights and biases of each layer after 1 step.

        Args:
            learning_rate (int): The learning rate of the training.
        """
        for layer in self.layers:
            layer.weights.set_value((layer.weights - learning_rate * layer.dW).eval())
            layer.biases.set_value((layer.biases - learning_rate * layer.db).eval())

    def __get_cost_value(self, Y_hat, Y):
        """Gets the cost value of the training.

        Args:
            Y_hat (np.ndarray): The observed output of the network.
            Y (np.ndarray): The correct output.
        Returns:
            np.ndarray: The cost of each output element in matrix.
        """
        m = Y_hat.shape[1]
        cost = -1 / m * (T.dot(Y, T.log(Y_hat).T) + T.dot(T.sub(1, Y), T.log(1 - Y_hat).T))
        return T.squeeze(cost)

    def __get_accuracy_value(self, Y_hat, Y):
        """Gets the accuracy of the outputs.

        Args:
            Y_hat (np.ndarray): The observed output of the network.
            Y (np.ndarray): The correct output.
        Returns:
            np.ndarray: the total accuracy of the training.
        """
        Y_hat_ = self.__convert_prob_into_class(Y_hat)
        return T.eq(Y_hat_, Y).all(axis=0).mean()

    def __convert_prob_into_class(self, probs):
        """Converts the probabilities of the outputs to output classes.

        Args:
            probs (np.ndarray): The probabilities of the outputs.
        Returns:
            np.ndarray: The classes of the outputs.
        """
        probs = T.set_subtensor(probs[probs > 0.5], 1)
        return T.set_subtensor(probs[probs <= 0.5], 0)

    def __getstate__(self):
        """Gets the state of the object to be pickled by Python.

        Returns:
            (list, int): The list of layers and the best loss value.
        """
        return (self.layers, self.best_loss)

    def __setstate__(self, state):
        """Sets the state of the object from pickled information.

        Args:
            state (list, int): The state of the pickled object.
        """
        l, bl = state
        self.layers = l
        self.best_loss = bl