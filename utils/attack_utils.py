import numpy as np
import keras.backend as K
import tensorflow as tf

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS


def linf_loss(X1, X2):
    return np.max(np.abs(X1 - X2), axis=(1, 2, 3))


def gen_adv_loss(logits, y, loss='logloss', mean=False):
    """
    Generate the loss function.
    """

    if loss == 'training':
        # use the model's output instead of the true labels to avoid
        # label leaking at training time
        y = K.cast(K.equal(logits, K.max(logits, 1, keepdims=True)), "float32")
        y = y / K.sum(y, 1, keepdims=True)
        out = K.categorical_crossentropy(y, logits, from_logits=True)
    elif loss == 'logloss':
        out = K.categorical_crossentropy(y, logits, from_logits=True)
    else:
        raise ValueError("Unknown loss: {}".format(loss))

    if mean:
        out = K.mean(out)
    # else:
    #     out = K.sum(out)
    return out


def gen_grad(x, logits, y, loss='logloss'):
    """
    Generate the gradient of the loss function.
    """

    adv_loss = gen_adv_loss(logits, y, loss)

    # Define gradient of loss wrt input
    grad = K.gradients(adv_loss, [x])[0]
    return grad

def gen_grad_ens(x, logits, y):

    adv_loss = K.categorical_crossentropy(logits[0], y, from_logits=True)
    if len(logits) >= 1:
        for i in range(1, len(logits)):
            adv_loss += K.categorical_crossentropy(logits[i], y, from_logits=True)

    grad = K.gradients(adv_loss, [x])[0]
    return adv_loss, grad

def gen_grad_cw(x, logits, y, kappa=100.0):
    real = tf.reduce_sum(y*logits[0], 1)
    other = tf.reduce_max((1-y)*logits[0] - (y*10000), 1)
    loss = tf.maximum(0.0,real-other+kappa)
    if len(logits) >= 1:
        for i in range(1, len(logits)):
            real = tf.reduce_sum(y*logits[i], 1)
            other = tf.reduce_max((1-y)*logits[i] - (y*10000), 1)
            loss += tf.maximum(0.0,real-other+args.kappa)
    grad = -1.0 * K.gradients(loss, [x])[0]
    return grad
