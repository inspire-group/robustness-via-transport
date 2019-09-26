import numpy as np
import tensorflow as tf
import math 
import keras
from keras.models import *
from keras.datasets import mnist, cifar10

import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import scipy.misc as misc


def two_class_convert(X_train, Y_train, X_test, Y_test, class_1, class_2, dataset='MNIST', no_of_examples=1000):
    NUM_CLASSES = 2
    class_1 = class_1
    class_2 = class_2
    # if dataset == 'fMNIST':
    #     Y_train_uncat = np.argmax(Y_train,axis=1)
    #     Y_test_uncat = np.argmax(Y_test,axis=1)
    # else:
    Y_train_uncat = Y_train
    Y_test_uncat = Y_test
    c1_idx = np.where(Y_train_uncat==class_1)
    c2_idx = np.where(Y_train_uncat==class_2)

    X_c1 = X_train[c1_idx][:no_of_examples]
    X_c2 = X_train[c2_idx][:no_of_examples]

    Y_train = np.zeros((2*no_of_examples,2))
    Y_train[:no_of_examples][:,0] = 1
    Y_train[no_of_examples:][:,1] = 1

    X_train = np.vstack((X_c1,X_c2))

    # Getting 3/7 from test
    c1_idx = np.where(Y_test_uncat==class_1)
    c2_idx = np.where(Y_test_uncat==class_2)

    X_c1 = X_test[c1_idx]
    X_c2 = X_test[c2_idx]

    test_len_3_7 = len(X_c1)+len(X_c2)

    Y_test = np.zeros((test_len_3_7,2))
    Y_test[:len(X_c1)][:,0] = 1
    Y_test[len(X_c1):][:,1] = 1

    X_test = np.vstack((X_c1,X_c2))

    return X_train, Y_train, X_test, Y_test


def two_class_convert_cifar(X, Y, class_1, class_2, no_of_examples=1000):
    NUM_CLASSES = 2
    class_1 = class_1
    class_2 = class_2

    c1_idx = np.where(Y==class_1)
    c2_idx = np.where(Y==class_2)

    X_c1 = X[c1_idx][:no_of_examples]
    X_c2 = X[c2_idx][:no_of_examples]

    # Y_train = np.zeros((2*no_of_examples,2))
    # Y_train[:no_of_examples][:,0] = 1
    # Y_train[no_of_examples:][:,1] = 1

    Y_c1 = Y[c1_idx][:no_of_examples]
    Y_c2 = Y[c2_idx][:no_of_examples]

    # print Y_train

    X = np.vstack((X_c1,X_c2))
    Y = np.hstack((Y_c1,Y_c2))

    print Y.shape

    return X,Y