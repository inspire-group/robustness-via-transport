import numpy as np
import argparse
import time
import os

from utils.datasets import data_mnist
from keras.datasets import cifar10

import scipy.spatial.distance
# from scipy.optimize import linear_sum_assignment
from utils.hungarian import linear_sum_assignment

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from mpl_toolkits.axes_grid1 import ImageGrid

def save_adv_images(cm, indices_1, indices_2, X_c1, X_c2,eps):
	adv_indices = np.where(cm==0.0)
	X_1 = X_c1[indices_1[adv_indices]]
	X_2 = X_c2[indices_1[adv_indices]]
	adv_images = (X_1 + X_2)/2.0
	no_to_print = len(adv_images)
	print no_to_print
	if 'MNIST' in args.dataset:
		adv_images = adv_images.reshape((no_to_print,28,28))
	elif 'CIFAR-10' in args.dataset:
		adv_images = adv_images.reshape((no_to_print,32,32,3))
	fig = plt.figure(1, (4., 4.))
	nrows = int(no_to_print/10.0) + 1
	grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(nrows, 10),  # creates 2x2 grid of axes
                 axes_pad=0.0,  # pad between axes in inch.
                 label_mode='1')

	for i in range(no_to_print):
		if i < no_to_print:
			if 'MNIST' in args.dataset:
				grid[i].imshow(adv_images[i],cmap='gray')
			elif 'CIFAR-10' in args.dataset:
				grid[i].imshow(adv_images[i])
	image_dir = 'images'
	if not os.path.exists(image_dir):
		os.makedirs(image_dir)
	image_dir += '/'
	image_name = image_dir + args.dataset + '_' + str(class_1) + '_' + str(class_2) + '_' + args.norm + '_' + str(eps) + '.png'
	plt.savefig(image_name)	

def data_setup():
	if 'MNIST' in args.dataset:
		IMAGE_ROWS = 28
		IMAGE_COLS = 28
		NUM_CHANNELS = 1
		DATA_DIM = IMAGE_ROWS*IMAGE_COLS*NUM_CHANNELS
		NUM_CLASSES = 10
		X_train, Y_train, X_test, Y_test = data_mnist(args.dataset)
		X_train = X_train.reshape((len(X_train),DATA_DIM))
		X_test = X_test.reshape((len(X_test),DATA_DIM))
		print('Loaded f/MNIST data')
	elif args.dataset == 'CIFAR-10':
		IMAGE_ROWS = 32
		IMAGE_COLS = 32
		NUM_CHANNELS = 3
		DATA_DIM = IMAGE_ROWS*IMAGE_COLS*NUM_CHANNELS
		NUM_CLASSES = 10
		(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

		X_train = X_train.reshape((len(X_train),DATA_DIM))
		X_test = X_test.reshape((len(X_test),DATA_DIM))

		Y_train = Y_train.reshape((len(X_train)))
		Y_test = Y_test.reshape((len(X_test)))

		X_train = X_train.astype('float32')
		X_test = X_test.astype('float32')
		X_train /= 255.
		X_test /= 255.
		print('Loaded CIFAR-10 data')

	return X_train, Y_train, X_test, Y_test


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='MNIST',
                    help="dataset to be used")
parser.add_argument("--norm", default='l2',
                    help="norm to be used")
parser.add_argument("--no_of_examples", type=int, default=200,
                    help="norm to be used")

args = parser.parse_args()

X_train, Y_train, X_test, Y_test = data_setup()

print X_train.shape
print Y_train.shape

if 'MNIST' in args.dataset or 'CIFAR-10' in args.dataset:
	no_of_examples = args.no_of_examples
	class_1 = 3
	class_2 = 7
	dist_dir = 'distances'
	if not os.path.exists(dist_dir):
		os.makedirs(dist_dir)
	dist_dir += '/'
	dist_mat_name = dist_dir + args.dataset + '_' + str(class_1) + '_' + str(class_2) + '_' + str(no_of_examples) + '_' + args.norm + '.npy'
	c1_idx = np.where(Y_train==class_1)
	c2_idx = np.where(Y_train==class_2)
	X_c1 = X_train[c1_idx][:no_of_examples]
	X_c2 = X_train[c2_idx][:no_of_examples]
	if os.path.exists(dist_mat_name):
		D_12 = np.load(dist_mat_name)
	else:
		if args.norm == 'l2':
			D_12 = scipy.spatial.distance.cdist(X_c1,X_c2,metric='euclidean')
		elif args.norm == 'linf':
			D_12 = scipy.spatial.distance.cdist(X_c1,X_c2,metric='chebyshev')
		np.save(dist_mat_name,D_12)
	print D_12

if args.norm == 'l2' and 'MNIST' in args.dataset:
	eps_list = np.linspace(0.0,5.0,21)
	print('List of eps for %s norm: %s' % (args.norm, eps_list))
elif args.norm == 'l2' and 'CIFAR-10' in args.dataset:
	eps_list = np.linspace(4.0,10.0,13)
	print('List of eps for %s norm: %s' % (args.norm, eps_list))
elif args.norm == 'linf' and 'MNIST' in args.dataset:
	eps_list = np.linspace(0.1,0.5,5)
	print('List of eps for %s norm: %s' % (args.norm, eps_list))
elif args.norm == 'linf' and 'CIFAR-10' in args.dataset:
	eps_list = np.linspace(0.1,0.5,5)
	print('List of eps for %s norm: %s' % (args.norm, eps_list))

save_file_name = str(class_1) + '_' + str(class_2) + '_' + str(no_of_examples) + '_' + args.dataset + '_' + args.norm

result_dir = 'results'
if not os.path.exists(result_dir):
	os.makedirs(result_dir)
result_dir += '/'
f = open(result_dir + save_file_name + '.txt', 'a')
f.write('eps,cost,inf_loss'+'\n')

marked_curr = np.zeros((no_of_examples, no_of_examples), dtype=int)
for eps in eps_list:
	cost_matrix = D_12 > 2*eps
	cost_matrix = cost_matrix.astype(float)
	# print cost_matrix
	match_dir = 'matchings'
	if not os.path.exists(match_dir):
		os.makedirs(match_dir)
	match_dir += '/'
	curr_file_name = match_dir + save_file_name + '_{0:.1f}.npy'.format(eps)

	if os.path.exists(curr_file_name):
		output = np.load(curr_file_name)
		output = np.expand_dims(output,axis=0)
	else:
		time1 = time.time()
		if eps>eps_list[0]:
			output = linear_sum_assignment(cost_matrix, marked_curr, False)
		else:
			output = linear_sum_assignment(cost_matrix, None, True)
	
		np.save(curr_file_name, output[0])

		marked_curr = output[1]

		time2 = time.time()

		print('Time taken for %s examples per class for eps %s is %s' % (no_of_examples, eps, time2-time1))

	raw_cost = np.float(cost_matrix[output[0][0], output[0][1]].sum())

	costs = cost_matrix[output[0][0], output[0][1]]
	save_adv_images(costs, output[0][0], output[0][1], X_c1, X_c2,eps)

	mean_cost = raw_cost/(no_of_examples)

	min_error = (1-mean_cost)/2

	print('At eps %s, cost: %s ; inf error: %s' % (eps, mean_cost, min_error)) 

	f.write(str(eps)+','+str(mean_cost)+','+str(min_error) + '\n')

f.close()