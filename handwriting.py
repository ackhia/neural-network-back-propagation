
import numpy as np

from datareaders.handwriting import get_data_set
from nn.bp import Network

if __name__ == '__main__':
  np.random.seed(0)

  data_training = get_data_set()
  data_test = get_data_set('test')
  print('Data loaded..')

  n = Network([28*28, 20, 20, 10])

  cost_before = n.cost(data_test)
  print(f'cost_before training={cost_before}')

  n.train_network(data_training, 10)

  cost_after = n.cost(data_test)
  print(f'cost_after training={cost_after}')

  actual = int(data_test[0][-1])
  predicted = np.argmax(n.predict(data_test[0][:-1]))
  print(f'actual={actual} predicted={predicted}')



