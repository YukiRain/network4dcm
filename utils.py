import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
<<<<<<< HEAD


_kernel_init = np.array([[1,2,3,2,1],
                         [2,5,6,5,2],
                         [3,6,8,6,3],
                         [2,5,6,5,2],
                         [1,2,3,2,1]]).astype(np.float32)
_kernel_gaussian = _kernel_init / float(_kernel_init.sum())
=======
import pydensecrf.densecrf as dcrf
>>>>>>> CRF and mysterious network changing...


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

<<<<<<< HEAD
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

=======
>>>>>>> CRF and mysterious network changing...
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

<<<<<<< HEAD
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

=======
>>>>>>> CRF and mysterious network changing...
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
<<<<<<< HEAD
    return 1.0 - np.square(decision - ys).sum() / float(area)
=======
    return 1.0 - np.square(decision - ys).sum() / float(area)


'''
    以下CRF内容高能预警！！！以下CRF内容高能预警！！！以下CRF内容高能预警！！！
    以下CRF内容高能预警！！！以下CRF内容高能预警！！！以下CRF内容高能预警！！！
    以下CRF内容高能预警！！！以下CRF内容高能预警！！！以下CRF内容高能预警！！！
'''

_kernel_5x5 = np.array([[1,2,3,2,1],
                        [2,5,6,5,2],
                        [3,6,8,6,3],
                        [2,5,6,5,2],
                        [1,2,3,2,1]], dtype=np.float32) / 84.0
_kernel_3x3 = np.array([[0.0751136, 0.123841, 0.0751136],
                        [0.123841, 0.20418, 0.123841],
                        [0.0751136, 0.123841, 0.0751136]], dtype=np.float32)
_sobel_x = np.array([[-3, 0, 3],
                     [-10, 0, 10],
                     [-3, 0, 3]], dtype=np.float32)
_sobel_y = _sobel_x.transpose()

def conv2d(input_op, n_out, kh=3, kw=3, dh=1, dw=1, name='conv2d'):
    # No activation, no bias, only applied in CRF
    n_in = input_op.get_shape()[-1].value
    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel_w',
                                 shape=(kh, kw, n_in, n_out),
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        return conv

def appearance_kernel(input_op):
    n_in = input_op.get_shape()[-1].value
    kernel_x = np.array([_sobel_x for _ in range(n_in)], dtype=np.float32).transpose((1, 2, 0))[:, :, :, None]
    kernel_y = np.array([_sobel_y for _ in range(n_in)], dtype=np.float32).transpose((1, 2, 0))[:, :, :, None]
    tensor_x = tf.constant(kernel_x, dtype=tf.float32, name='sobel_x')
    tensor_y = tf.constant(kernel_y, dtype=tf.float32, name='sobel_y')

    conv_x = tf.nn.conv2d(input_op, tensor_x, strides=(1,1,1,1), padding='SAME')
    conv_y = tf.nn.conv2d(input_op, tensor_y, strides=(1, 1, 1, 1), padding='SAME')
    grad = tf.square(conv_x / 2) + tf.square(conv_y / 2)
    return tf.exp(-grad)

def message_passing(factor_a, factor_b, img_op, kernel_1x1=None):
    # Using factor_b to update factor_a and return factor_a
    entity = tf.concat([factor_a, factor_b], axis=3)
    appear = appearance_kernel(img_op)

    kernel_3x3 = np.array([_kernel_3x3, _kernel_3x3], dtype=np.float32).transpose((1, 2, 0))[:, :, :, None]
    kernel_5x5 = np.array([_kernel_5x5, _kernel_5x5], dtype=np.float32).transpose((1, 2, 0))[:, :, :, None]
    kernel_1 = tf.constant(kernel_3x3, dtype=tf.float32, name='g_3x3')
    kernel_2 = tf.constant(kernel_5x5, dtype=tf.float32, name='g_5x5')

    conv_a_1 = tf.nn.conv2d(entity, kernel_1, strides=(1, 1, 1, 1), padding='SAME') - factor_b
    conv_a_2 = tf.nn.conv2d(entity, kernel_2, strides=(1, 1, 1, 1), padding='SAME') - factor_b
    conv_a_3 = appear * factor_a - appear * factor_b - factor_b

    inf_a = tf.concat([conv_a_1, conv_a_2, conv_a_3], axis=3)
    weighted = tf.nn.conv2d(inf_a, kernel_1x1, strides=(1, 1, 1, 1), padding='SAME')
    return weighted

# The idea is borrowed from https://github.com/DrSleep/tensorflow-deeplab-resnet/tree/crf
def dense_crf(probs, img=None, n_iters=10, n_classes=2,
              sxy_gaussian=(1, 1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.

    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).

    Returns:
      Refined predictions after MAP inference.
    """
    _, h, w, _ = probs.shape

    probs = probs[0].transpose(2, 0, 1).copy(order='C')  # Need a contiguous array.

    d = dcrf.DenseCRF2D(w, h, n_classes)  # Define DenseCRF model.
    U = -np.log(probs + 1e-8)  # Unary potential (avoid probs underflow to 0)
    U = U.reshape((n_classes, -1))  # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        assert (img.shape[1: 3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)
>>>>>>> CRF and mysterious network changing...
