import os
import gzip
import numpy as np

from keras.datasets import mnist
import scipy.io as sio
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from keras.utils import np_utils

def load_fmnist(path, kind='train'):


    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def data_mnist(dataset,one_hot=False):
    """
    Preprocess MNIST dataset
    """
    # the data, shuffled and split between train and test sets

    IMAGE_ROWS = 28
    IMAGE_COLS = 28
    NUM_CHANNELS = 1
    DATA_DIM = IMAGE_ROWS*IMAGE_COLS*NUM_CHANNELS
    NUM_CLASSES = 10
    if dataset == 'MNIST':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif dataset == 'fMNIST':
        X_train, y_train = load_fmnist('/home/data', kind='train')
        X_test, y_test = load_fmnist('/home/data', kind='t10k')


    X_train = X_train.reshape(X_train.shape[0],
                              IMAGE_ROWS,
                              IMAGE_COLS,
                              NUM_CHANNELS)

    X_test = X_test.reshape(X_test.shape[0],
                            IMAGE_ROWS,
                            IMAGE_COLS,
                            NUM_CHANNELS)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if one_hot:
        # convert class vectors to binary class matrices
        y_train = np_utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
        y_test = np_utils.to_categorical(y_test, NUM_CLASSES).astype(np.float32)

    return X_train, y_train, X_test, y_test
