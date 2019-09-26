import os 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import keras.backend as K
import cPickle as pickle
from os.path import basename
from time import time
from keras.utils import np_utils

from utils.mnist import load_model
from utils.utils_2c import two_class_convert
from utils.datasets import data_mnist
from utils.fgs import symbolic_fgs, iter_fgs, symbolic_fg, iter_fg
from utils.attack_utils import gen_grad, gen_grad_ens, gen_grad_cw
from utils.tf_utils import tf_test_error_rate, batch_eval


BATCH_SIZE = 64
NUM_CLASSES = 10
IMAGE_ROWS = 28
IMAGE_COLS = 28
NUM_CHANNELS = 1

SAVE_FLAG = False

def main(attack, src_model_names, target_model_name):
    np.random.seed(0)
    tf.set_random_seed(0)

    X_train, Y_train, X_test, Y_test = data_mnist(args.dataset)

    if args.two_class:
        NUM_CLASSES = 2
        class_1 = 3
        class_2 = 7
        X_train, Y_train, X_test, Y_test = two_class_convert(X_train, Y_train, X_test, Y_test, class_1, class_2, args.dataset)
        # X_test = X_train
        # Y_test = Y_train

    dim = IMAGE_ROWS*IMAGE_COLS*NUM_CHANNELS

    x = K.placeholder((None,
                       IMAGE_ROWS,
                       IMAGE_COLS,
                       NUM_CHANNELS))

    y = K.placeholder((None, NUM_CLASSES))

    # source model for crafting adversarial examples
    src_models = [None] * len(src_model_names)
    for i in range(len(src_model_names)):
        src_models[i] = load_model(src_model_names[i])

    src_model_name_joint = ''
    for i in range(len(src_models)):
        src_model_name_joint += basename(src_model_names[i])

    # model(s) to target
    if target_model_name is not None:
        target_model = load_model(target_model_name)

    # simply compute test error
    if attack == "test":
        for (name, src_model) in zip(src_model_names, src_models):
            _, _, err = tf_test_error_rate(src_model, x, X_test, Y_test)
            print '{}: {:.1f}'.format(basename(name), err)
        if target_model_name is not None:
            _, _,err = tf_test_error_rate(target_model, x, X_test, Y_test)
            print('{}: {:.1f}'.format(basename(target_model_name), err))

        return        
    
    if args.targeted_flag == 1:
        pickle_name =  attack + '_' + src_model_name_joint+'_'+'_'+args.loss_type+'_targets.p'
        if os.path.exists(pickle_name):
            targets = pickle.load(open(pickle_name,'rb'))
        else:
            targets = []
            allowed_targets = list(range(NUM_CLASSES))
            for i in range(len(Y_test)):
                allowed_targets.remove(Y_test_uncat[i])
                targets.append(np.random.choice(allowed_targets))
                allowed_targets = list(range(NUM_CLASSES))
            # targets = np.random.randint(10, size = BATCH_SIZE*BATCH_EVAL_NUM)
            targets = np.array(targets)
            print targets
            targets_cat = np_utils.to_categorical(targets, NUM_CLASSES).astype(np.float32)
            Y_test = targets_cat
            if SAVE_FLAG == True:
                pickle.dump(Y_test, open(pickle_name,'wb'))
    
    
    # take the random step in the RAND+FGSM
    if attack == "rand_fgs":
        X_test = np.clip(
            X_test + args.alpha * np.sign(np.random.randn(*X_test.shape)),
            0.0, 1.0)
        eps -= args.alpha

    logits = [None] * len(src_model_names)
    for i in range(len(src_model_names)):
        curr_model = src_models[i]
        logits[i] = curr_model(x)

    if args.loss_type == 'xent':
        loss, grad = gen_grad(x, logits, y)
    elif args.loss_type == 'cw':
        grad = gen_grad_cw(x, logits, y)
    if args.targeted_flag == 1:
        grad = -1.0 * grad

    for eps in eps_list:
        # FGSM and RAND+FGSM one-shot attack
        if attack in ["fgs", "rand_fgs"] and args.norm == 'linf':
            adv_x = symbolic_fgs(x, grad, eps=eps)
        elif attack in ["fgs", "rand_fgs"] and args.norm == 'l2':
            adv_x = symbolic_fg(x, grad, eps=eps)

        # iterative FGSM
        if attack == "ifgs":
            l=500
            X_test = X_test[0:l]
            Y_test = Y_test[0:l]

            adv_x = x
            # iteratively apply the FGSM with small step size
            for i in range(args.num_iter):
                adv_logits = [None] * len(src_model_names)
                for i in range(len(src_model_names)):
                    curr_model = src_models[i]
                    adv_logits[i] = curr_model(adv_x)

                if args.loss_type == 'xent':
                    loss, grad = gen_grad_ens(adv_x, adv_logits, y)
                elif args.loss_type == 'cw':
                    grad = gen_grad_cw(adv_x, adv_logits, y)
                if args.targeted_flag == 1:
                    grad = -1.0 * grad

                adv_x = symbolic_fgs(adv_x, grad, args.delta, True)
                r = adv_x - x
                r = K.clip(r, -eps, eps)
                adv_x = x+r

            adv_x = K.clip(adv_x, 0, 1)  
        if attack == "ifg":
            l=2000
            X_test = X_test[0:l]
            Y_test = Y_test[0:l]
            if args.noise: 
                noise_delta = np.random.normal(size=(l,784))
                norm_vec = np.linalg.norm(noise_delta,axis=1)
                noise_delta /= np.expand_dims(norm_vec,1)
                noise_delta = noise_delta.reshape((l,28,28,1))
                X_test = X_test + args.eps*noise_delta
                X_test = np.clip(X_test, 0, 1) # ensure valid pixel range 
            adv_x = iter_fg(src_models[0],x,y,args.num_iter,args.delta,eps,args.targeted_flag)

        pickle_name = attack + '_' + src_model_name_joint+'_'+args.loss_type+'_'+str(eps)+'_adv.p'
        if args.targeted_flag == 1:
            pickle_name = attack + '_' + src_model_name_joint+'_'+args.loss_type+'_'+str(eps)+'_adv_t.p'

        if os.path.exists(pickle_name):
            print 'Loading adversarial samples'
            X_adv = pickle.load(open(pickle_name,'rb'))
        else:
            print 'Generating adversarial samples'
            X_adv = batch_eval([x, y], [adv_x], [X_test, Y_test])[0]
            if SAVE_FLAG == True:
                pickle.dump(X_adv, open(pickle_name,'wb'))

        avg_l2_perturb = np.mean(np.linalg.norm((X_adv-X_test).reshape(len(X_test),dim),axis=1))

        # white-box attack
        l = len(X_adv)
        # print ('Carrying out white-box attack')
        for (name, src_model) in zip(src_model_names, src_models):
            preds_adv, orig, err = tf_test_error_rate(src_model, x, X_adv, Y_test[0:l])
            if args.targeted_flag==1:
                err = 100.0 - err
            print '{}->{}: {:.1f}'.format(basename(name), basename(name), err)

        # black-box attack
        if target_model_name is not None:
            print ('Carrying out black-box attack')
            preds, _, err = tf_test_error_rate(target_model, x, X_adv, Y_test)
            if args.targeted_flag==1:
                err = 100.0 - err
            print '{}->{}: {:.1f}, {}, {} {}'.format(src_model_name_joint, basename(target_model_name), err, avg_l2_perturb, eps, attack)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("attack", help="name of attack",
                        choices=["test", "fgs", "ifgs", "ifg", "rand_fgs", "CW_ens"])
    parser.add_argument('src_models', nargs='*',
                        help="source models for attack")
    parser.add_argument('--target_model',type=str,
                        help='path to target model(s)')
    parser.add_argument("--dataset", default='MNIST',
                    help="dataset to be used")
    parser.add_argument("--eps", type=float, default=None,
                        help="FGS attack scale")
    parser.add_argument("--loss_type", type=str, default='cw',
                        help="Type of loss to use")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="RAND+FGSM random perturbation scale")
    parser.add_argument("--delta", type=float, default=0.01,
                        help="Iterated FGS step size")
    parser.add_argument("--num_iter", type=int, default=40,
                        help="Iterated FGS step size")
    parser.add_argument("--kappa", type=float, default=100.0,
                        help="CW attack confidence")
    parser.add_argument("--norm", type=str, default='linf',
                        help="Norm to use for attack")
    parser.add_argument("--targeted_flag", type=int, default=0,
                        help="Carry out targeted attack")
    parser.add_argument("--two_class", action='store_true')
    parser.add_argument("--noise", action='store_true')


    args = parser.parse_args()

    if args.eps is None:
        if args.norm == 'linf':
            # eps_list = list(np.linspace(0.025, 0.1, 4))
            # eps_list.extend(np.linspace(0.15, 0.5, 8))
            eps_list = [0.3]
            if args.attack == "ifgs":
                eps_list = [0.3]
        elif args.norm == 'l2':
            if args.attack == 'ifg':
                eps_list = [3.8]
                # args.delta = eps_list[0]/args.num_iter
            else:
                eps_list = list(np.linspace(0.0, 2.0, 5))
                eps_list.extend(np.linspace(2.5, 9.0, 14))
            # eps_list = [5.0]
    else:
        eps_list = []
        eps_list.append(args.eps)
    print(eps_list)

    main(args.attack, args.src_models, args.target_model)