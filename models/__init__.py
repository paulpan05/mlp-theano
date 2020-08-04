import theano
import theano.tensor as T

class Sequential:
    def __init__(self):
        self.layers = []
        self.best_loss = None
    def add(self, layer):
        if len(self.layers) == 0:
            if not layer.n_inputs:
                raise Exception('Need to have n_inputs for layer.')
        else:
            layer.n_inputs = self.layers[-1].units
        self.layers.append(layer)
    def compile(self):
        for layer in self.layers:
            layer._Dense__load()
    def fit(self, X, Y, epochs, learning_rate=0.1):
        for i in range(epochs):
            Y_hat = self.__feedforward(X)
            cost = self.__get_cost_value(Y_hat, Y)
            if self.best_loss == None or cost < self.best_loss:
                self.best_loss = cost
            accuracy = self.__get_accuracy_value(Y_hat, Y)
            self.__backprop(X, Y_hat, Y)
            self.__update(learning_rate)
    def __feedforward(self, X):
        A = X
        for layer in self.layers:
            layer._Dense__forward(A)
            A = layer.A
        return A
    def __backprop(self, X, Y_hat, Y):
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape.eval())
        dA = - ((Y / Y_hat) - ((1 - Y) / (1 - Y_hat)))
        i = len(self.layers) - 1
        while i >= 0:
            A_prev = None
            if i == 0:
                A_prev = X
            else:
                A_prev = self.layers[i-1].A
            self.layers[i]._Dense__backward(dA, A_prev)
            dA = self.layers[i].dA
            i -= 1
    def __update(self, learning_rate):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.dW
            layer.biases -= learning_rate * layer.db
    def __get_cost_value(self, Y_hat, Y):
        m = Y_hat.shape[1]
        cost = -1 / m * (T.dot(Y, T.log(Y_hat).T) + T.dot(T.sub(1, Y), T.log(1 - Y_hat).T))
        return T.squeeze(cost)
    def __get_accuracy_value(self, Y_hat, Y):
        Y_hat_ = self.__convert_prob_into_class(Y_hat)
        return T.eq(Y_hat_, Y).all(axis=0).mean()
    def __convert_prob_into_class(self, probs):
        probs = T.set_subtensor(probs[probs > 0.5], 1)
        return T.set_subtensor(probs[probs <= 0.5], 0)
    def __getstate__(self):
        return (self.layers, self.best_loss)
    def __setstate__(self, state):
        l, bl = state
        self.layers = l
        self.best_loss = bl