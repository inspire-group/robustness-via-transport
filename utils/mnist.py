from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

import argparse
import numpy as np

BATCH_SIZE = 64
NUM_CLASSES = 10
IMAGE_ROWS = 28
IMAGE_COLS = 28
NUM_CHANNELS = 1

def modelA():
    NUM_CLASSES = 2
    model = Sequential()
    model.add(Conv2D(64, (5, 5),
                            padding='valid'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    return model

def modelA_large():
    print('Model A large')
    NUM_CLASSES = 2
    model = Sequential()
    model.add(Conv2D(256, (5, 5),
                            padding='valid'))
    model.add(Activation('relu'))

    model.add(Conv2D(256, (5, 5)))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    return model


def modelB():
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(IMAGE_ROWS,
                                        IMAGE_COLS,
                                        NUM_CHANNELS)))
    model.add(Convolution2D(64, 8, 8,
                            subsample=(2, 2),
                            border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 6, 6,
                            subsample=(2, 2),
                            border_mode='valid'))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 5, 5,
                            subsample=(1, 1)))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(NUM_CLASSES))
    return model


def modelC():
    model = Sequential()
    model.add(Convolution2D(128, 3, 3,
                            border_mode='valid',
                            input_shape=(IMAGE_ROWS,
                                         IMAGE_COLS,
                                         NUM_CHANNELS)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    return model


def modelD():
    model = Sequential()

    model.add(Flatten(input_shape=(IMAGE_ROWS,
                                   IMAGE_COLS,
                                   NUM_CHANNELS)))

    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, init='he_normal', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES))
    return model

def modelE():
    model = Sequential()

    model.add(Flatten(input_shape=(IMAGE_ROWS,
                                   IMAGE_COLS,
                                   NUM_CHANNELS)))

    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))

    model.add(Dense(NUM_CLASSES))

    return model

def modelF():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3,
                            border_mode='valid',
                            input_shape=(IMAGE_ROWS,
                                         IMAGE_COLS,
                                         NUM_CHANNELS)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))

    model.add(Dense(NUM_CLASSES))

    return model

def data_gen_mnist(X_train):
    datagen = ImageDataGenerator()

    datagen.fit(X_train)
    return datagen

def model_mnist(type=0):
    """
    Defines MNIST model using Keras sequential model
    """

    models = [modelA, modelA_large, modelB, modelC, modelD, modelE, modelF]

    return models[type]()

def load_model(model_path, type=0):

    try:
        with open(model_path+'.json', 'r') as f:
            json_string = f.read()
            model = model_from_json(json_string)
    except IOError:
        model = model_mnist(type=type)

    model.load_weights(model_path)
    return model
