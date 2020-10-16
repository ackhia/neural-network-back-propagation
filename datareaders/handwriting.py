

import struct
import numpy as np

def chunk(l, n):
  for i in range(0, len(l), n):  
    yield l[i:i + n] 

def get_data_set(type='training'):
  assert (type == 'training' or type == 'test')
  if type == 'training':
    data_file_name = 'train-images-idx3-ubyte'
    label_file_name = 'train-labels-idx1-ubyte'
  else:
    data_file_name = 't10k-images-idx3-ubyte'
    label_file_name = 't10k-labels-idx1-ubyte'

  with open(f'./data/{data_file_name}', 'rb') as image_files:
    images_bin = image_files.read()

  with open(f'./data/{label_file_name}', 'rb') as labels_file:
    lables_bin = labels_file.read()

  lables_header = struct.unpack('>II', lables_bin[:8])

  assert lables_header[0] == 2049
  label_count = lables_header[1]
  lables = struct.unpack('B'*label_count, lables_bin[8:])

  image_header = struct.unpack('>IIII', images_bin[:16])

  assert image_header[0] == 2051
  image_count = image_header[1]
  assert image_count == label_count
  width = image_header[2]
  height = image_header[2]
  images_unpacked = struct.unpack('B' * image_count * width * height, images_bin[16:])
  images = list(chunk(images_unpacked, width * height))

  data = [np.concatenate((np.array(i) / 255, np.array([l]))) for (l, i) in zip(lables, images)]
  return data

def display_image(image):
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg

  plt.imshow(tuple(chunk(image, 28)))
  plt.show()

if __name__ == '__main__':
  images = get_data_set()
  print(images[0][-1])
  display_image(images[0][:-1])