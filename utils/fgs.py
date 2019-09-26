import keras.backend as K
from attack_utils import gen_grad, gen_grad_cw
import tensorflow as tf

def symbolic_fgs(x, grad, eps=0.3, clipping=True):
    """
    FGSM attack.
    """

    # signed gradient
    normed_grad = K.sign(grad)

    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = K.stop_gradient(x + scaled_grad)

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)
    return adv_x

def symbolic_fg(x, grad, eps=0.3, clipping=True):
    """
    FG attack
    """
    # Unit vector in direction of gradient
    reduc_ind = list(xrange(1, len(x.get_shape())))
    normed_grad = grad / tf.sqrt(tf.reduce_sum(tf.square(grad),
                                                   reduction_indices=reduc_ind,
                                                   keepdims=True))
    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = K.stop_gradient(x + scaled_grad)

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)

    return adv_x


def iter_fgs(model, x, y, steps, alpha, eps, clipping=True):
    """
    I-FGSM attack.
    """

    adv_x = x
    # iteratively apply the FGSM with small step size
    for i in range(steps):
        logits = model(adv_x)
        grad = gen_grad(adv_x, logits, y)

        adv_x = symbolic_fgs(adv_x, grad, alpha, True)
        r = adv_x - x
        r = K.clip(r, -eps, eps)
        adv_x = x+r

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)


    return adv_x

def iter_fg(model, x, y, steps, alpha, eps, targeted=0, clipping=True):
    """
    I-FGSM attack.
    """

    adv_x = x
    # iteratively apply the FGSM with small step size
    for i in range(steps):
        logit_list = []
        logits = model(adv_x)
        logit_list.append(logits)
        grad = gen_grad_cw(adv_x, logit_list, y)
        if targeted == 1:
            grad = -1.0 * grad

        adv_x = symbolic_fg(adv_x, grad, alpha, True)
        r = adv_x - x
        reduc_ind = list(xrange(1, len(r.get_shape())))
        norm_r = tf.sqrt(tf.reduce_sum(tf.square(r),
                                                   reduction_indices=reduc_ind,
                                                   keepdims=True))
        normed_r = r / norm_r
        r = tf.minimum(norm_r,eps) * normed_r
        adv_x = x+r

    if clipping:
        adv_x = K.clip(adv_x, 0, 1)


    return adv_x
