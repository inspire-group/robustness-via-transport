import keras
from keras import backend as K
# from tensorflow.python.platform import flags
from keras.models import save_model
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

from utils.tf_utils import tf_train, tf_test_error_rate
from utils.mnist import *
from utils.utils_2c import two_class_convert
from utils.datasets import data_mnist

BATCH_SIZE = 64
NUM_CLASSES = 10
IMAGE_ROWS = 28
IMAGE_COLS = 28
NUM_CHANNELS = 1

def main(model_name, model_type):
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"

    X_train, Y_train, X_test, Y_test = data_mnist(args.dataset)

    if args.two_class:
        NUM_CLASSES = 2
        class_1 = 3
        class_2 = 7
        X_train, Y_train, X_test, Y_test = two_class_convert(X_train, Y_train, X_test, Y_test, class_1, class_2)

    data_gen = data_gen_mnist(X_train)

    x = K.placeholder((None,
                       IMAGE_ROWS,
                       IMAGE_COLS,
                       NUM_CHANNELS
                       ))

    y = K.placeholder(shape=(None, NUM_CLASSES))
    print NUM_CLASSES

    model = model_mnist(type=model_type)

    # print(model.summary())

    # Train an MNIST model
    tf_train(x, y, model, X_train, Y_train, data_gen, args.epochs, None, None)

    # Finally print the result!
    _, _, test_error = tf_test_error_rate(model, x, X_test, Y_test)
    print('Test error: %.1f%%' % test_error)
    save_model(model, model_name)
    json_string = model.to_json()
    with open(model_name+'.json', 'wr') as f:
        f.write(json_string)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to model")
    parser.add_argument("--dataset", default='MNIST',
                    help="dataset to be used")
    parser.add_argument("--type", type=int, help="model type", default=0)
    parser.add_argument("--epochs", type=int, default=6, help="number of epochs")
    parser.add_argument("--two_class", action='store_true')
    args = parser.parse_args()

    main(args.model, args.type)
