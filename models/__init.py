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
    def compile(self, learning_rate=None):
        for layer in layers:
            layer.__load()
    def fit(self, X, y, epochs, learning_rate=0.1):
        for i in range(epochs):
            pass
    def __get_cost_value(self, Y_hat, Y):
        m = Y_hat.shape[1]
        cost = -1 / m * (T.dot(Y, T.log(Y_hat).T) + T.dot(1 - Y, T.log(1 - Y_hat).T))
        return T.squeeze(cost)
    def __get_accuracy_value(self, Y_hat, Y):
        Y_hat_ = self.__convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()
    def __convert_prob_into_class(self, probs):
        return T.set_subtensor(probs[probs > 0.5], 1).set_subtensor(probs[probs <= 0.5], 0)
    def __getstate__(self):
        return (self.layers, self.best_loss)
    def __setstate__(self, state):
        l, bl = state
        self.layers = l
        self.best_loss = bl