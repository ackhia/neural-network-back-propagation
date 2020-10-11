
import numpy as np
import pandas
from datareaders import handwriting

class Network():
  def __init__(self, shape):
    self.shape = shape
    self.layers = []
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
      activations = np.dot(layer['weights'], self.layers[i-1]['activations'])
      activations_bias = np.add(activations, layer['biases'])

      #Apply relu
      activations_bias[activations_bias<0] = 0.0
      
      #Squash the last layer
      if i == len(self.layers)-1:
        activations_bias = activations_bias / np.amax(activations_bias)
        pass

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
          next_layer_weights = self.layers[i+1]['weights'][:,j]
          errors.append(np.sum(next_layer_weights * self.layers[i+1]['deltas']))
      
      layer['deltas'] = [e * self.relu_derivative(a) for e, a in zip(errors, layer['activations'])]
        
if __name__ == "__main__":
  n = Network([28*28, 40, 40, 10])
  dataset = handwriting.get_data_set('test')
  print('Data set read..')
  n.predict(dataset[0]['data'])
  n.backward_propergate_error(dataset[0]['label'])
  print(n.cost(dataset))