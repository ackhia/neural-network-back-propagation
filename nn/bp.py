
import numpy as np
import pandas


class Network():
    def __init__(self, shape, learning_rate=0.1, weights=None):
        self.shape = shape
        self.layers = []
        self.learning_rate = learning_rate

        for i, v in enumerate(shape):
            layer = {}
            layer['activations'] = np.empty(v)
            if i != 0:
                layer['biases'] = np.full(v, 0.1)
                if weights:
                    layer['weights'] = weights[i]
                else:
                    layer['weights'] = np.random.rand(v, shape[i-1]) - 0.5
            self.layers.append(layer)

    def predict(self, input):
        self.layers[0]['activations'] = input
        for i, layer in enumerate(self.layers[1:], 1):
            activations = np.dot(
                layer['weights'], self.layers[i-1]['activations'])
            activations_bias = np.add(activations, layer['biases'])

            activations_bias = 1.0 / (1.0 + np.exp(-activations_bias))

            layer['activations'] = activations_bias

        return self.layers[-1]['activations']

    def cost(self, dataset):
        sum = 0.0
        for image in dataset:
            actual = self.predict(image[:-1])
            expected = np.zeros(self.shape[-1])
            expected[int(image[-1])] = 1.0
            for i, v in enumerate(actual):
                sum += np.square(v - expected[i])
        return sum / len(dataset)

    def sigmoid_derivative(self, n):
        return n * (1.0 - n)

    def backward_propergate_error(self, expected):
        for i, layer in zip(range(len(self.layers)-1, 0, -1), self.layers[:0:-1]):
            if i == len(self.layers) - 1:
                errors = expected - layer['activations']
            else:
                errors = []
                for j in range(len(layer['weights'])):
                    next_layer_weights = self.layers[i+1]['weights'][:, j]
                    errors.append(np.sum(next_layer_weights *
                                         self.layers[i+1]['deltas']))

            layer['deltas'] = [
                e * self.sigmoid_derivative(a) for e, a in zip(errors, layer['activations'])]

    def update_weights_biases(self):
        for i, layer in enumerate(self.layers[1:], 1):
            for neuron in range(len(layer['weights'])):
                layer['weights'][neuron] = layer['weights'][neuron] + self.learning_rate * \
                    layer['deltas'][neuron] * self.layers[i-1]['activations']
                layer['biases'][neuron] = self.learning_rate * layer['deltas'][neuron]

    def train_network(self, dataset, epoch_count):
        for epoch in range(epoch_count):
            sum_error = 0.0

            for row in dataset:
                outputs = self.predict(row[:-1])
                expected = np.zeros(self.shape[-1])
                expected[int(row[-1])] = 1.0
                for i, v in enumerate(outputs):
                    sum_error += np.square(v - expected[i])
                self.backward_propergate_error(expected)
                self.update_weights_biases()
            print(f'Epoch={epoch}, error_rate={sum_error}')

