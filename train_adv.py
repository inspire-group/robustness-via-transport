import os 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras import backend as K
from keras.models import save_model
import tensorflow as tf

from utils.mnist import *
from utils.datasets import data_mnist
from utils.tf_utils import tf_train, tf_test_error_rate
from utils.attack_utils import gen_grad, gen_grad_cw
from utils.utils_2c import two_class_convert
from utils.fgs import symbolic_fgs, iter_fgs, symbolic_fg, iter_fg
from os.path import basename

def main(model_name, adv_model_names, model_type):
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"

    X_train, Y_train, X_test, Y_test = data_mnist(args.dataset)
    # Get MNIST test data
    if args.two_class:
        NUM_CLASSES = 2
        class_1 = 3
        class_2 = 7
        X_train, Y_train, X_test, Y_test = two_class_convert(X_train, Y_train, X_test, Y_test, class_1, class_2, args.dataset)

    data_gen = data_gen_mnist(X_train)

    x = K.placeholder(shape=(None,
                             IMAGE_ROWS,
                             IMAGE_COLS,
                             NUM_CHANNELS))

    y = K.placeholder(shape=(BATCH_SIZE, NUM_CLASSES))

    eps = args.eps
    norm = args.norm

    # if src_models is not None, we train on adversarial examples that come
    # from multiple models
    adv_models = [None] * len(adv_model_names)
    ens_str = ''
    for i in range(len(adv_model_names)):
        adv_models[i] = load_model(adv_model_names[i])
	if len(adv_models)>0:
	    name = basename(adv_model_names[i])
	    model_index = name.replace('model','')
	    ens_str += model_index
    model = model_mnist(type=model_type)

    x_advs = [None] * (len(adv_models) + 1)

    for i, m in enumerate(adv_models + [model]):
        if args.norm == 'linf':
            if args.iter == 0:
                logits = m(x)
                grad = gen_grad(x, logits, y, loss='training')
                x_advs[i] = symbolic_fgs(x, grad, eps=eps)
            elif args.iter == 1:
                x_advs[i] = iter_fgs(m, x, y, steps = 40, alpha = 0.01, eps = args.eps)
        elif args.norm == 'l2':
            if args.iter == 0:
                logits = m(x)
                grad = gen_grad(x, logits, y, loss='training')
                x_advs[i] = symbolic_fg(x, grad, eps=eps)
            elif args.iter == 1:
                x_advs[i] = iter_fg(m, x, y, args.num_iter,args.delta,eps)

    # Train an MNIST model
    if args.noise:
        noise_delta = np.random.normal(size=(len(X_train),784))
        norm_vec = np.linalg.norm(noise_delta,axis=1)
        noise_delta /= np.expand_dims(norm_vec,1)
        noise_delta *= np.expand_dims(np.random.uniform(0.0,args.eps,len(X_train)),axis=1)
        noise_delta = noise_delta.reshape((len(X_train),28,28,1))
        X_train = X_train + noise_delta
    X_train = np.clip(X_train, 0, 1) # ensure valid pixel range 
    tf_train(x, y, model, X_train, Y_train, data_gen, args.epochs, x_advs=x_advs, benign = args.ben)

    # Finally print the result!
    preds_adv, orig, err = tf_test_error_rate(model, x, X_test, Y_test)
    print('Test error: %.1f%%' % err)
    model_name += '_' + args.dataset + '_' + str(eps) + '_' + str(norm) + '_' + str(args.epochs) + '_' + ens_str
    if args.iter == 1:
        model_name += 'iter'
        model_name += '_' + str(args.num_iter)
        model_name += '_' + str(args.delta)
    if args.ben == 0:
        model_name += '_nob'
    if args.two_class:
        model_name += '_' + str(class_1) + '_' + str(class_2)
    if args.noise:
        model_name += '_noise'
    save_model(model, model_name)
    json_string = model.to_json()
    with open(model_name+'.json', 'wr') as f:
        f.write(json_string)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to model")
    parser.add_argument('adv_models', nargs='*',
                        help='path to adv model(s)')
    parser.add_argument("--dataset", default='MNIST',
                    help="dataset to be used")
    parser.add_argument("--type", type=int, help="model type", default=1)
    parser.add_argument("--epochs", type=int, default=12,
                        help="number of epochs")
    parser.add_argument("--eps", type=float, default=0.3,
                        help="FGS attack scale")
    parser.add_argument("--norm", type=str, default='linf',
                        help="norm used to constrain perturbation")
    parser.add_argument("--iter", type=int, default=0,
                        help="whether an iterative training method is to be used")
    parser.add_argument("--ben", type=int, default=0,
                        help="whether benign data is to be used while performing adversarial training")
    parser.add_argument("--delta", type=float, default=0.01,
                        help="Iterated FGS step size")
    parser.add_argument("--num_iter", type=int, default=40,
                        help="Iterated FGS step size")
    parser.add_argument("--kappa", type=float, default=100.0,
                        help="CW attack confidence")
    parser.add_argument("--two_class", action='store_true')
    parser.add_argument("--noise", action='store_true')

    args = parser.parse_args()
    main(args.model, args.adv_models, args.type)
