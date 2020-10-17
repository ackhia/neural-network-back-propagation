
import numpy as np

from datareaders.handwriting import get_data_set
from nn.bp import Network

def print_accuracy(n, data):
  correct = 0
  incorrect = 0
  for d in data:
    output = n.predict(d[:-1])
    result = np.argmax(output)
    if result == d[-1]:
      correct += 1
    else:
      incorrect += 1

  print(f'correct={correct} incorrect={incorrect} percent={(1-incorrect/correct)*100:.2f}%')


if __name__ == '__main__':
  np.random.seed(0)

  data_training = get_data_set()
  data_test = get_data_set('test')
  print('Data loaded..')

  n = Network([28*28, 40, 40, 10])

  cost_before = n.cost(data_test)
  print(f'cost_before training={cost_before}')

  n.train_network(data_training, 10)

  cost_after = n.cost(data_test)
  print(f'cost_after training={cost_after}')

  actual = int(data_test[0][-1])
  predicted = np.argmax(n.predict(data_test[0][:-1]))
  print(f'actual={actual} predicted={predicted}')

  print_accuracy(n, data_test)

