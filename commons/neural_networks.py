from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout, GaussianNoise
from tensorflow.keras.layers import Conv1D, Conv2D, AveragePooling2D, MaxPooling1D, MaxPooling2D, BatchNormalization, AveragePooling1D
from tensorflow.keras.models import Sequential


class NeuralNetwork:

    def __init__(self):
        pass

    def mlp_random(self, classes, number_of_samples, activation, neurons, layers, learning_rate):
        model = Sequential()
        model.add(BatchNormalization(input_shape=(number_of_samples,)))
        for l_i in range(layers):
            model.add(Dense(neurons, activation=activation, kernel_initializer='he_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_random(self, classes, number_of_samples, activation, neurons, conv_layers, filters, kernel_size, stride, layers, learning_rate):
        model = Sequential()
        for layer_index in range(conv_layers):
            if layer_index == 0:
                model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation='relu', padding='valid',
                                 input_shape=(number_of_samples, 1)))
            else:
                model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation='relu', padding='valid'))

        model.add(Flatten())
        for layer_index in range(layers):
            model.add(Dense(neurons, activation=activation, kernel_initializer='random_uniform', bias_initializer='zeros'))

        model.add(Dense(classes, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

        return model
