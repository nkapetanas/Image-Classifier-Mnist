import numpy as np
np.random.seed(1400)
import keras
from keras.datasets import mnist


NUMBER_OF_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 0.01

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape data
x_train = x_train.reshape(60000, 784) # 28 x 28 = 784 in order to have all pixel values to one big vector
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)



