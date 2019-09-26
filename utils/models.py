# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf

class WideResnet(object):
  """
  ResNet model.
  
  Refernce: https://github.com/MadryLab/cifar10_challenge
  # Input: Include per-image standardization step thus input could follow any of [0, 225] or [0,1] range. 
  """

  def __init__(self, mode, input_tensor, n_class=10, scale_factor=1):
    """ResNet constructor.
    Args:
      mode: One of 'train' and 'eval'.
    """
    self.mode = mode
    self.input_tensor= input_tensor
    self.n_class= n_class
    self.scale_factor= scale_factor
    self._build_model()
  

  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    assert self.mode == 'train' or self.mode == 'eval'
    """Build the core model within the graph."""
    with tf.variable_scope('input',reuse=tf.AUTO_REUSE):

      self.x_input = self.input_tensor

      self.y_input = tf.placeholder(tf.int64, shape=None)


      input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                               self.x_input)
      x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))



    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    res_func = self._residual

    # Uncomment the following codes to use w28-10 wide residual network.
    # It is more memory efficient than very deep residual network and has
    # comparably good performance.
    # https://arxiv.org/pdf/1605.07146v1.pdf
    filters = [16, 16*self.scale_factor, 32*self.scale_factor, 64*self.scale_factor]


    # Update hps.num_residual_units to 9

    with tf.variable_scope('unit_1_0', reuse=tf.AUTO_REUSE):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in range(1, 5):
      with tf.variable_scope('unit_1_%d' % i, reuse=tf.AUTO_REUSE):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0', reuse=tf.AUTO_REUSE):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in range(1, 5):
      with tf.variable_scope('unit_2_%d' % i, reuse=tf.AUTO_REUSE):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0', reuse=tf.AUTO_REUSE):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in range(1, 5):
      with tf.variable_scope('unit_3_%d' % i, reuse=tf.AUTO_REUSE):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last', reuse=tf.AUTO_REUSE):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, 0.1)
      self.x_fc = self._global_avg_pool(x)

    with tf.variable_scope('logit', reuse=tf.AUTO_REUSE):
      self.pre_softmax = self._fully_connected_with_no_bias(self.x_fc, self.n_class)

    self.predictions = tf.argmax(self.pre_softmax, 1)
    self.correct_prediction = tf.equal(self.predictions, self.y_input)
    self.num_correct = tf.reduce_sum(
        tf.cast(self.correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(
        tf.cast(self.correct_prediction, tf.float32))

    with tf.variable_scope('costs', reuse=tf.AUTO_REUSE):
      self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.pre_softmax, labels=self.y_input)
      self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
      self.mean_xent = tf.reduce_mean(self.y_xent)
      self.weight_decay_loss = self._decay()

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.name_scope(name):
      return tf.contrib.layers.batch_norm(
          inputs=x,
          decay=.9,
          center=True,
          scale=True,
          activation_fn=None,
          updates_collections=None,
          is_training=(self.mode == 'train'))

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation', reuse=tf.AUTO_REUSE):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation', reuse=tf.AUTO_REUSE):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)

    with tf.variable_scope('sub1', reuse=tf.AUTO_REUSE):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2', reuse=tf.AUTO_REUSE):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, 0.1)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add', reuse=tf.AUTO_REUSE):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _fully_connected_with_no_bias(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    return tf.matmul(x, w)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])




######################################################################################################################
class DenseNet():
  """
  DenseNet-BC model.
  
  TensorFlow implementation of https://github.com/facebookresearch/odin/blob/master/code/densenet.py
  # Input: Include per-image standardization step thus input could follow any of [0, 225] or [0,1] range. 
  """

  def __init__(self, mode, input_tensor, depth=100, n_class=10, growth_rate=12, reduction=0.5):
    """DenseNet-BC constructor.
    Args:
      mode: One of 'train' and 'eval'.
    """
    self.mode = mode
    self.depth = depth
    self.input_tensor = input_tensor
    self.n_class = n_class
    self.growth_rate = growth_rate
    self.reduction = reduction
    ##
    # create additional 
    ##
    self._build_model()

  def _build_model(self):
    assert self.mode == 'train' or self.mode == 'eval'
    """Build the core model within the graph."""
    in_planes = 2 * self.growth_rate
    n = int(int((self.depth - 4) / 3)/2)

    with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
      self.x_input = self.input_tensor
      self.y_input = tf.placeholder(tf.int64, shape=None)

      input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                               self.x_input)

    block = self.BottleneckBlock
    x = self._conv('init_conv', input_standardized, 3, 3, in_planes, [1, 1, 1, 1])
    # 1st block
    with tf.variable_scope('block_0', reuse=tf.AUTO_REUSE):
      x = self.DenseBlock(x, n, in_planes, self.growth_rate, block) # write it
      in_planes = int(in_planes + n * self.growth_rate)
      x = self.TransitionBlock(x, in_planes, int(math.floor(in_planes*self.reduction))) #write it
      in_planes = int(math.floor(in_planes * self.reduction)) 
    # 2nd block
    with tf.variable_scope('block_1', reuse=tf.AUTO_REUSE):
      x = self.DenseBlock(x, n, in_planes, self.growth_rate, block) # write it
      in_planes = int(in_planes + n * self.growth_rate)
      x = self.TransitionBlock(x, in_planes, int(math.floor(in_planes*self.reduction))) #write it
      in_planes = int(math.floor(in_planes * self.reduction)) 
    #3rd block
    with tf.variable_scope('block_2', reuse=tf.AUTO_REUSE):
      x = self.DenseBlock(x, n, in_planes, self.growth_rate, block) # write it
      in_planes = int(in_planes + n * self.growth_rate)

    with tf.variable_scope('final_layers', reuse=tf.AUTO_REUSE):
      x = self._batch_norm('bn_0', x)
      x = self._relu(x)
      x = tf.nn.avg_pool(x, [1, 8, 8, 1], [1, 8, 8, 1], padding='SAME')
      self.x_fc = tf.reshape(x, (-1, in_planes))

    with tf.variable_scope('logit', reuse=tf.AUTO_REUSE):
      self.pre_softmax = self._fully_connected_with_no_bias(self.x_fc, self.n_class) 

    self.predictions = tf.argmax(self.pre_softmax, 1)
    self.correct_prediction = tf.equal(self.predictions, self.y_input)
    self.num_correct = tf.reduce_sum(
        tf.cast(self.correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(
        tf.cast(self.correct_prediction, tf.float32))

    with tf.variable_scope('costs', reuse=tf.AUTO_REUSE):
      self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.pre_softmax, labels=self.y_input)
      self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
      self.mean_xent = tf.reduce_mean(self.y_xent)
      self.weight_decay_loss = self._decay()

  def BottleneckBlock(self, x, in_planes, out_planes):
    orig_x = x
    inter_planes = out_planes * 4
    x = self._batch_norm('bn_0', x)
    x = self._relu(x)
    x = self._conv('conv_0', x, 3, in_planes, inter_planes, [1, 1, 1, 1])
    x = self._batch_norm('bn_1', x)
    x = self._relu(x)
    x = self._conv('conv_1', x, 3, inter_planes, out_planes, [1, 1, 1, 1])
    return tf.concat([orig_x, x], axis=-1)

  def DenseBlock(self, x, nb_layers, in_planes, growth_rate, block):
    for i in range(nb_layers):
      with tf.variable_scope('denseblock_%d' % i, reuse=tf.AUTO_REUSE):
        x = block(x, in_planes+i*growth_rate, growth_rate)
    return x

  def TransitionBlock(self, x, in_planes, out_planes):
    with tf.variable_scope('transitionblock', reuse=tf.AUTO_REUSE):
      x = self._batch_norm('bn_0', x)
      x = self._relu(x)
      x = self._conv('conv_0', x, 1, in_planes, out_planes, [1, 1, 1, 1])
      x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
      return x

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.name_scope(name):
      return tf.contrib.layers.batch_norm(
          inputs=x,
          decay=.9,
          center=True,
          scale=True,
          activation_fn=None,
          updates_collections=None,
          is_training=(self.mode == 'train'))

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _fully_connected_with_no_bias(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    return tf.matmul(x, w)
