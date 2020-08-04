class Sequential:
    def __init__(self):
        self.layers = []
    def add(self, layer):
        if len(self.layers) == 0:
            if not layer.n_inputs:
                raise Exception('Need to have n_inputs for layer.')
        else:
            layer.n_inputs = self.layers[-1].units
        self.layers.append(layer)
    def __getstate__(self):
        return self.layers
    def __setstate__(self, state):
        self.layers = state