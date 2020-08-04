import pickle
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
    def fit(self, X, Y, epochs, batch_size=32, learning_rate=0.01):
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
            layer.weights = (layer.weights - learning_rate * layer.dW).eval()
            layer.biases = (layer.biases - learning_rate * layer.db).eval()
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