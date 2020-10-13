
import numpy as np
import pandas
from datareaders import handwriting


class Network():
    def __init__(self, shape, learning_rate=0.1):
        self.shape = shape
        self.layers = []
        self.learning_rate = learning_rate

        for i, v in enumerate(shape):
            layer = {}
            layer['activations'] = np.empty(v)
            if i != 0:
                layer['biases'] = np.full(v, 0.1)
                layer['weights'] = np.random.rand(v, shape[i-1]) - 0.5
            self.layers.append(layer)

    def predict(self, input):
        self.layers[0]['activations'] = input
        for i, layer in enumerate(self.layers[1:], 1):
            activations = np.dot(
                layer['weights'], self.layers[i-1]['activations'])
            activations_bias = np.add(activations, layer['biases'])

            # Apply relu
            activations_bias[activations_bias < 0] = 0.0

            # Squash the last layer
            """if i == len(self.layers)-1:
                max_activation = np.amax(activations_bias)
                if max_activation > 0:
                    activations_bias = activations_bias / \
                        np.amax(activations_bias)"""

            layer['activations'] = activations_bias

        return self.layers[-1]['activations']

    def cost(self, dataset):
        sum = 0.0
        for image in dataset:
            actual = self.predict(image['data'])
            expected = np.zeros(self.shape[-1])
            expected[image['label']] = 1.0
            for i, v in enumerate(actual):
                sum += np.square(v - expected[i])
        return sum / len(dataset)

    def relu_derivative(self, n):
        return 1.0 if n > 0 else 0.0

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
                e * self.relu_derivative(a) for e, a in zip(errors, layer['activations'])]

    def update_weights_biases(self):
        for i, layer in enumerate(self.layers[1:], 1):
            for neuron in range(len(layer['weights'])):
                layer['weights'][neuron] = layer['weights'][neuron] + self.learning_rate * \
                    layer['deltas'][neuron] * self.layers[i-1]['activations']

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
            print(f'Epoch: {epoch}, error_rate {sum_error}')


dataset = np.array([[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]])

if __name__ == "__main__":
    n = Network([2, 2, 2])
    n.train_network(dataset, 200)
