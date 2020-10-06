
import numpy as np

class Network():
  def __init__(self, shape):
    self.layers = []
    for i, v in enumerate(shape):
      layer = {}
      layer['activations'] = np.empty(v)
      if i != 0:
        layer['biases'] = np.full(v,0)
        layer['weights'] = np.array([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]])#np.full((v, shape[i-1]), 0.5)
      self.layers.append(layer)

  def predict(self, input):
    self.layers[0]['activations'] = input
    for i, layer in enumerate(self.layers[1:], 1):
      activation_matrix = np.array([self.layers[i-1]['activations'],]*len(layer['activations'])).transpose()
      activations = np.dot(layer['weights'], activation_matrix)
      layer['activations'] = np.add(activations, layer['biases'])
    
    return self.layers[-1]['activations'][:,0]

if __name__ == "__main__":
  n = Network([4, 2])
  input = np.array([0.1,0.2,0.3,0.4])
  print(n.predict(input))