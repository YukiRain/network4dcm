import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim


_kernel_init = np.array([[1,2,3,2,1],
                         [2,5,6,5,2],
                         [3,6,8,6,3],
                         [2,5,6,5,2],
                         [1,2,3,2,1]]).astype(np.float32)
_kernel_gaussian = _kernel_init / float(_kernel_init.sum())


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, input_op, is_training=True):
        return tf.contrib.layers.batch_norm(input_op,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=is_training,
                                            scope=self.name)


def show_all_variables():
    all_variables = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(all_variables, print_info=True)

def leaky_relu(input_op, leak=0.2, name='linear'):
    return tf.maximum(input_op, leak*input_op, name=name)

def conv2d(input_op, n_out, name, kh=3, kw=3, dh=1, dw=1):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel_w',
                                 shape=(kh, kw, n_in, n_out),
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(0.0))
        return tf.nn.bias_add(conv, biases)

def conv2d_relu(input_op, n_out, name, kh=3, kw=3, dh=1, dw=1, activate='relu'):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel_w',
                                 shape=(kh, kw, n_in, n_out),
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(0.0))
        z_out = tf.nn.bias_add(conv, biases)
        if activate == 'relu':
            return tf.nn.relu(z_out, name='relu')
        elif activate == 'lrelu':
            return leaky_relu(z_out)
        else:
            return z_out

def weighted_conv2d_with_gaussian(input_op, n_out, name):
    # This function is for message passing
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name):
        kernel = tf.convert_to_tensor(_kernel_gaussian, dtype=tf.float32, name='fixed_gaussian')
        conv = tf.nn.conv2d(input_op, kernel, strides=(1, 1, 1, 1), padding='SAME')

        weights = tf.get_variable(name='weights_w',
                                  shape=(1, 1, n_in, n_out),
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        return tf.nn.conv2d(conv, weights, strides=(1,1,1,1), padding='SAME')

def pooling(input_op, name, kh=2, kw=2, dh=2, dw=2, pooling_type='max'):
    if 'max' in pooling_type:
        return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)
    else:
        return tf.nn.avg_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)

def deconv2d(input_op, output_shape, kh=3, kw=3, dh=2, dw=2, name='deconv', bias_init=0.0):
    n_in = input_op.get_shape()[-1].value
    n_out = output_shape[-1]
    # filter : [height, width, output_channels, in_channels]
    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernels',
                                 shape=(kh, kw, n_out, n_in),
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        deconv = tf.nn.conv2d_transpose(input_op, kernel,
                                        output_shape=output_shape,
                                        strides=(1, dh, dw, 1))
        biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(bias_init))
        return tf.nn.relu(tf.nn.bias_add(deconv, biases), name='deconv_activate')

def fully_connect(input_op, n_out, name='fully_connected', bias_init=0.0, activate='lrelu', with_kernels=False):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name='matrix',
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

        biases = tf.get_variable(name='bias', shape=(n_out), initializer=tf.constant_initializer(bias_init))
        return tf.matmul(input_op, kernel) + biases

def inception_block(input_op, n_out, name='inception_block'):
    # the proportion of n_out is 1:2:1
    with tf.variable_scope(name):
        conv1x1 = conv2d(input_op, n_out=n_out / 4, name='conv1x1', kh=1, kw=1)
        conv3x3 = conv2d(input_op, n_out=n_out / 2, name='conv3x3', kh=3, kw=3)
        conv5x5 = conv2d(input_op, n_out=n_out / 4, name='conv5x5', kh=5, kw=5)
        concatenated = tf.concat([conv1x1, conv3x3, conv5x5], axis=3)
        return tf.nn.relu(concatenated)

def get_accuracy(xs, ys):
    # xs and ys should have the same shape/size
    # ys must be normalized to 0-1
    decision = np.zeros_like(xs, dtype=np.float32)
    decision[xs > 0.5] = 1.0
    decision[xs < 0.5] = 0.0
    area = xs.shape[0] * xs.shape[1]
    return 1.0 - np.square(decision - ys).sum() / float(area)