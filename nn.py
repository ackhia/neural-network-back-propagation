
import numpy as np

class Network():
  def __init__(self, shape):
    self.layers = []
    for i, v in enumerate(shape):
      layer = {}
      layer['activations'] = np.empty(v)
      layer['biases'] = np.random.rand(v)
      if i != 0:
        layer['weights'] = np.random.rand(v, shape[i-1])
      self.layers.append(layer)

if __name__ == "__main__":
  n = Network([50, 20, 20, 5])