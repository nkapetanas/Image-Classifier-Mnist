import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

np.random.seed(1400)
from keras.datasets import mnist

NUMBER_OF_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.01

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape data
x_train = x_train.reshape(60000, 784)  # 28 x 28 = 784 in order to have all pixel values to one big vector
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 0 - 255 to 0-1
x_test /= 255


def get_model_relu_fiveHL(activation_function):
    model = Sequential()
    model.add(Dense(32, activation=activation_function, input_shape=(784,)))
    model.add(Dense(32, activation=activation_function))
    model.add(Dense(32, activation=activation_function))
    model.add(Dense(32, activation=activation_function))
    model.add(Dense(32, activation=activation_function))
    model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))  # output layer

    return model


# convert class vectors to binary class matrices -> one-hot-vector
y_train = keras.utils.to_categorical(y_train, NUMBER_OF_CLASSES)

y_test = keras.utils.to_categorical(y_test, NUMBER_OF_CLASSES)

# model = get_model_relu_fiveHL('relu')
# model = get_model_relu_fiveHL('tanh')
model = get_model_relu_fiveHL('sigmoid')

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=LEARNING_RATE), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)
print('Loss: ' + str(score[0]) + 'Accuracy: ' + str(score[1]))
