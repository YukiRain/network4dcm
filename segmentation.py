from datetime import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from dcm_read import dcm_reader
from cnn_read import Reader

network_path = './deep_learning/network/'

batch_size = 1

# 共25层，前12层与后12层完全对称，最后一层为输出层
class FCNet(object):
    def __init__(self, img_size=512, learning_rate=0.001):
        self.paras = dict()
        self.sess = tf.InteractiveSession()
        self.xin = tf.placeholder(tf.float32, [None, img_size**2])
        self.y_ = tf.placeholder(tf.float32, [None, img_size**2])

        self.x_img = tf.reshape(self.xin, shape=(-1, img_size, img_size, 1))
        self._learning_rate = learning_rate
        self._img_size = img_size

        # 初始化整个网络，返回最终结果的向量形式
        self.y_conv = self._inference_op()

        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.y_conv, self.y_), 2.0))
        self.optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self.cost)
        tf.global_variables_initializer().run()

    def _conv_op(self, input_op, n_out, para_list, name,
                 kh = 3, kw = 3, dh = 1, dw = 1):
        ''' parameter processing of VGG
        :param kh: 卷积核的宽度
        :param kw: 卷积核的高度
        :param n_out: 卷积核个数
        :param dh: 卷积层stride的宽度
        :param dw: 卷积层stride的高度
        :param para_list: 参数列表，list类型，用来装类中所有的变量
        :return: 使用ReLU激活后的输出，op类型
        '''
        n_in = input_op.get_shape()[-1].value

        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(name=scope + '_w',
                                     shape=[kh, kw, n_in, n_out],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())

            conv = tf.nn.conv2d(input_op, kernel, (1,dh,dw,1), padding='SAME')
            bias_init_val = tf.constant(0.1, shape=[n_out], dtype=tf.float32)
            biases = tf.Variable(bias_init_val, trainable=True, name='b')
            z_out = tf.nn.bias_add(conv, biases)
            activation = tf.nn.relu(z_out, name=scope)

            para_list[scope + '_w'] = kernel
            para_list[scope + '_b'] = biases
            return activation

    def _deconv_op(self ,input_op, output_shape, para_list, name,
                   kh = 3, kw = 3, dh = 2, dw = 2):
        n_in = input_op.get_shape()[-1].value
        n_out = output_shape[-1]
        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(name=scope + '_kernel',
                                     shape=[kh, kw, n_out, n_in],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            deconv = tf.nn.conv2d_transpose(value=input_op,
                                            filter=kernel,
                                            output_shape=output_shape,
                                            strides=[1,dh,dw,1],
                                            padding='SAME')
            bias_init_val = tf.constant(0.1, shape=[n_out], dtype=tf.float32)
            biases = tf.Variable(bias_init_val, trainable=True, name='b')
            z_out = tf.nn.bias_add(deconv, biases)
            activation = tf.nn.relu(z_out, name=scope)
            para_list[scope + '_kernel'] = kernel
            para_list[scope + '_bias'] = biases
            return activation

    def _fc_op(self, input_op, name, n_out, para_list):
        n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(name=scope+'_w',
                                     shape=[n_in, n_out],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32, name='b'))
            activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
            para_list[scope + '_w'] = kernel
            para_list[scope + '_b'] = biases
            return activation

    def _maxpool_op(self, input_op, name, kh = 2, kw = 2, dh = 2, dw = 2):
        return tf.nn.max_pool(input_op,
                              ksize=[1, kh, kw, 1],
                              strides=[1, dh, dw, 1],
                              padding='SAME',
                              name=name)

    def _inference_op(self):
        self.conv1_1 = self._conv_op(self.x_img, name='conv1_1', para_list=self.paras, n_out=32)
        self.conv1_2 = self._conv_op(self.conv1_1, name='conv1_2', para_list=self.paras, n_out=32)
        self.pool1 = self._maxpool_op(self.conv1_2, name='pool1')

        self.conv2_1 = self._conv_op(self.pool1, name='conv2_1', para_list=self.paras, n_out=128)
        self.conv2_2 = self._conv_op(self.conv2_1, name='conv2_2', para_list=self.paras, n_out=128)
        self.pool2 = self._maxpool_op(self.conv2_2, name='pool2')

        self.conv3_1 = self._conv_op(self.pool2, name='conv3_1', para_list=self.paras, n_out=512)
        self.conv3_2 = self._conv_op(self.conv3_1, name='conv3_2', para_list=self.paras, n_out=512)
        self.pool3 = self._maxpool_op(self.conv3_2, name='conv3_3')

        self.conv4_1 = self._conv_op(self.pool3, name='conv4_1', para_list=self.paras, n_out=512)
        self.conv4_2 = self._conv_op(self.conv4_1, name='conv4_2', para_list=self.paras, n_out=512)
        self.deconv4 = self._deconv_op(self.conv4_2, [batch_size,128,128,512], self.paras, name='deconv4')

        self.conv5_1 = self._conv_op(self.deconv4, name='conv5_1', para_list=self.paras, n_out=128)
        self.conv5_2 = self._conv_op(self.conv5_1, name='conv5_2', para_list=self.paras, n_out=128)
        self.deconv5 = self._deconv_op(self.conv5_2, [batch_size,256,256,128], self.paras, name='deconv5')

        self.conv6_1 = self._conv_op(self.deconv5, name='conv6_1', para_list=self.paras, n_out=32)
        self.conv6_2 = self._conv_op(self.conv6_1, name='conv6_2', para_list=self.paras, n_out=32)
        self.deconv6 = self._deconv_op(self.conv6_2, [batch_size,512,512,32], self.paras, name='deconv6')

        self.conv7_1 = self._conv_op(self.deconv6, name='conv7_1', para_list=self.paras, n_out=16)
        self.conv7_2 = self._conv_op(self.conv7_1, name='conv7_2', para_list=self.paras, n_out=1)
        return tf.reshape(self.conv7_2, shape=(-1, self._img_size**2))

    def train(self, reader, loop=20000, batch_num=5):
        loss_tab = list()
        for i in range(loop):
            batch_xs, batch_ys = reader.next_batch(batch_num)
            if i % 100 == 0:
                loss = self.cost.eval(feed_dict={self.xin: batch_xs, self.y_: batch_ys})
                print(datetime.now(), 'step %d, training loss: %g'  % (i, loss))
                loss_tab.append(loss)
            self.optimizer.run(feed_dict={self.xin : batch_xs, self.y_ : batch_ys})
        print('ALL DONE!!')
        return loss_tab

    # 输入必须是向量
    def predict(self, img):
        conv = self.conv7_2.eval(feed_dict={self.xin: img})
        return conv[0, :, :, 0]

    # Still require a vector as input and output accuracy
    def get_accuracy(self, img, label, thresh=100):
        conv = self.conv7_2.eval(feed_dict={self.xin: img})
        conv = conv[0, :, :, 0]
        comp = np.zeros_like(conv)
        comp[conv >= thresh] = 255
        return (((comp - label.reshape((self._img_size, self._img_size)))**2)/255).sum()/(self._img_size**2)

    def save(self, path):
        saver = tf.train.Saver()
        try:
            directory = path + 'fcnn.ckpt'
            save_path = saver.save(self.sess, directory)
            print('MODEL RESTORED IN: ' + save_path)
        except:
            print('SOMETHING WRONG...')

    def load(self, path=network_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path + 'fcnn.ckpt')
        print('LOAD FINISHED')


if __name__ == '__main__':
    reader = Reader()
    fcnn = FCNet(learning_rate=4e-4)
    fcnn.load(network_path)

    loss_tab = fcnn.train(reader, loop=2000, batch_num=batch_size)
    plt.figure()
    plt.plot(loss_tab)
    plt.show()

    if str(input('Save Model???')) == 'yy':
        fcnn.save(network_path)

    test_x, test_y = reader.get_test_data()
    print(test_x.shape, test_y.shape)
    for i in range(test_x.shape[0]):
        pred = fcnn.predict(test_x[i][None, :]).astype(np.uint)
        print(fcnn.get_accuracy(test_x[i][None, :], test_y[i]))
        plt.figure()
        plt.subplot(131)
        plt.imshow(pred, cmap='gray')
        plt.subplot(132)
        plt.imshow(test_y[i].reshape((512, 512)), cmap='gray')
        plt.subplot(133)
        plt.imshow(test_x[i].reshape((512, 512)), cmap='gray')
        plt.show()

    print('ALL DONE!!!')