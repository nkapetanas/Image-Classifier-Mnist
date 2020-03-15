import numpy as np

np.random.seed(1400)
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import matplotlib.pyplot as plt

from keras.datasets import mnist

NUMBER_OF_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 10
EPOCHS_C = 3
LEARNING_RATE = 0.01


def leCunActivationFunction(x):
    return 1.7159 * K.tanh(2 / 3 * x)


def create_plot_model_accuracy_keras(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def create_plot_model_loss_keras(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def create_plot_Keras_model(field):
    plt.plot(history.history[field])
    plt.title('model ' + field)
    plt.ylabel(field)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    plt.subplot(2, 1, 1)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')

    # plot accuracy
    plt.subplot(2, 1, 2)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.show()


def get_model(activation_function, totalNumberOfLayers):
    model = Sequential()
    model.add(Dense(32, activation=activation_function, input_shape=(784,)))

    for x in range(totalNumberOfLayers - 2):
        model.add(Dense(32, activation=activation_function))

    model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))  # output layer

    return model


# function to obtain grads for each parameter
def get_gradients(model, inputs, outputs):
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + weight)
    return np.array(output_grad)


def get_model_custom_activation(custom_activation, totalNumberOfLayers):
    get_custom_objects().update({'custom_activation': Activation(custom_activation)})
    model = Sequential()
    model.add(Dense(32, input_shape=(784,), activation=custom_activation))

    for x in range(totalNumberOfLayers - 2):
        model.add(Dense(32, activation=custom_activation))

    model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))  # output layer

    return model


def recall_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape data
x_train = x_train.reshape(60000, 784)  # 28 x 28 = 784 in order to have all pixel values to one big vector
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 0 - 255 to 0-1
x_test /= 255

# convert class vectors to binary class matrices -> one-hot-vector
y_train = keras.utils.to_categorical(y_train, NUMBER_OF_CLASSES)

y_test = keras.utils.to_categorical(y_test, NUMBER_OF_CLASSES)

######################### 5 Layer Models #############################
model = get_model('relu', 5)
# model = get_model('tanh', 5)
# model = get_model('sigmoid', 5)

######################### 20 Layer Models ############################
# model = get_model('relu', 20)
# model = get_model('tanh', 20)
# model = get_model('sigmoid', 20)

######################### 40 Layer Models ############################
# model = get_model('relu', 40)
# model = get_model('tanh', 40)
# model = get_model('sigmoid', 40)

######################### LeCun Model ####################
# model = get_model_custom_activation(leCunActivationFunction, 5)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=LEARNING_RATE),
              metrics=['accuracy', f1, precision_metric, recall_metric])

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                    validation_data=(x_test, y_test))

# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)

print('Loss: ' + str(loss) + ' Accuracy: ' + str(accuracy) + ' F1: ' + str(f1_score) + ' Precision: ' + str(
    precision) + ' Recall: ' + str(recall))

summarize_diagnostics(history)
print(history.history.keys())

create_plot_Keras_model('val_loss')
create_plot_Keras_model('val_accuracy')
create_plot_Keras_model('accuracy')

grads = get_gradients(model, x_train, y_train)

print(grads.shape)
for i, _ in enumerate(grads):
    print(grads[i].shape)
    max_gradient_layer_i = np.max(grads[i])
    print(max_gradient_layer_i)
    plt.scatter(i, max_gradient_layer_i, color='blue', marker='x')
    plt.title('Layer Depth vs. Max Gradient (LeCun)')
    plt.xlabel('Layer')
    plt.ylabel('Max Gradient')

plt.show()
