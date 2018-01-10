from datetime import datetime
import matplotlib.pyplot as plt
from functools import reduce
import os

from utils import *
from cnn_read import Reader

class CRF_RNN(object):
    def __init__(self, input_shape, input_dim, batch_size=4, learning_rate=0.0002,
                 model_dir='./checkpoint/', log_dir='.\\logs', pre_train=True):
        # Copy parameters
        # input_shape should be 3 dimensional list
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.model_dir = model_dir

        self._build_model()
        self.fcn_loss = self._loss_function()
        self.fcn_optim = tf.train.AdamOptimizer(learning_rate).minimize(self.fcn_loss, var_list=self.fcn_vars)
        # self.crf_optim = tf.train.AdamOptimizer(learning_rate).minimize(self.crf_loss, var_list=self.crf_vars)

        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # Initialize summary
        self.fcn_summary = tf.summary.scalar('fcn_loss', self.fcn_loss)
        # self.crf_summary = tf.summary.scalar('crf_loss', self.crf_loss)
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        # load pre_trained model if checkpoint is not empty
        if pre_train and len(os.listdir(self.model_dir)) != 0:
            _, self.counter = self.load()
        else:
            print('Build model from scratch!!')
            self.counter = 0

    def _build_model(self):
        self.input_image_vector = tf.placeholder(tf.float32, shape=(None, self.input_dim))
        self.input_image = tf.reshape(self.input_image_vector, shape=[self.batch_size] + self.input_shape)
        self.label_vector = tf.placeholder(tf.float32, shape=[None, self.input_dim])

        self.fcnn, self.fcnn_logits = self._build_fcn()
        # self.fc_crf = self._build_crf(self.fcnn, reuse=False)

        trainable_variables = tf.trainable_variables()
        self.fcn_vars = [var for var in trainable_variables if 'FCNN' in var.name]
        # self.crf_vars = [var for var in trainable_variables if 'CRF' in var.name]

    def _build_fcn(self):
        # deconv size
        row, col = self.input_shape[0], self.input_shape[1]
        row_p1, col_p1 = int(row / 2), int(col / 2)
        row_p2, col_p2 = int(row_p1 / 2), int(col_p1 / 2)

        with tf.variable_scope('FCNN'):
            conv1_1 = inception_block(self.input_image, n_out=32, name='inception1_1')
            conv1_2 = inception_block(conv1_1, n_out=32, name='conv1_2')
            pool_1 = pooling(conv1_2, name='pool_1')

            conv2_1 = inception_block(pool_1, n_out=128, name='conv2_1')
            conv2_2 = inception_block(conv2_1, n_out=128, name='conv2_2')
            pool_2 = pooling(conv2_2, name='pool_2')

            conv3_1 = inception_block(pool_2, n_out=512, name='conv3_1')
            conv3_2 = inception_block(conv3_1, n_out=512, name='conv3_2')
            pool_3 = pooling(conv3_2, name='pool_3')

            conv4_1 = inception_block(pool_3, n_out=512, name='conv4_1')
            deconv_1 = deconv2d(conv4_1, output_shape=[self.batch_size, row_p2, col_p2, 512], name='deconv_1')

            concat_1 = tf.concat([conv3_2, deconv_1], axis=3, name='concat_1')
            conv5_1 = inception_block(concat_1, n_out=128, name='conv5_1')
            deconv_2 = deconv2d(conv5_1, output_shape=[self.batch_size, row_p1, col_p1, 128], name='deconv_2')

            concat_2 = tf.concat([conv2_2, deconv_2], axis=3, name='concat_2')
            conv6_1 = inception_block(concat_2, n_out=32, name='conv6_1')
            conv6_2 = inception_block(conv6_1, n_out=32, name='conv6_2')
            deconv_3 = deconv2d(conv6_2, output_shape=[self.batch_size, row, col, 32], name='deconv_3')

            concat_3 = tf.concat([conv1_2, deconv_3], axis=3, name='concat_3')
            conv7_1 = inception_block(concat_3, n_out=32, name='conv7_1')
            conv7_2 = conv2d_relu(conv7_1, n_out=32, name='conv7_2')
            conv7_3 = conv2d(conv7_2, n_out=1, name='conv7_3')
            return tf.nn.sigmoid(conv7_3, name='sigmoid_fcn'), conv7_3

    def _build_crf(self, unary, reuse=False):
        # reuse: iteratively update Q_{i}
        with tf.variable_scope('CRFasRNN', reuse=reuse):
            pass

    def _loss_function(self):
        logit_vector = tf.reshape(self.fcnn_logits, shape=(self.batch_size, self.input_dim))
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_vector, labels=self.label_vector)
        return tf.reduce_sum(cross_entropy, name='loss_val')

    def train(self, reader, loop=20000, print_iter=100):
        for i in range(loop):
            batch_xs, batch_ys = reader.next_batch(self.batch_size)
            self.sess.run(self.fcn_optim, feed_dict={self.input_image_vector: batch_xs,
                                                     self.label_vector: batch_ys})
            # Log on screen
            if i % print_iter == 5:
                loss = self.sess.run(self.fcn_loss, feed_dict={self.input_image_vector: batch_xs,
                                                               self.label_vector: batch_ys})
                logging = ' --Iteration %d --FCN loss %g' % (i, loss)
                print(str(datetime.now()) + logging)
            # Log on tensorboard
            _, summary_str = self.sess.run([self.fcn_optim, self.fcn_summary],
                                           feed_dict={self.input_image_vector: batch_xs,
                                                      self.label_vector: batch_ys})
            self.writer.add_summary(summary_str, self.counter)
            self.counter += 1
        print('Training Finished!!')
        if 'y' in str(input('Save Model???')):
            self.save()

    def predict(self, imgvec, use_logits=False, as_list=False):
        if use_logits:
            pred = self.sess.run(self.fcnn_logits, feed_dict={self.input_image_vector: imgvec})
        else:
            pred = self.sess.run(self.fcnn, feed_dict={self.input_image_vector: imgvec})
        input_size = pred.shape[0]
        if as_list:
            return [pred[i, :, :, 0] for i in range(input_size)]
        else:
            return pred.reshape((input_size, self.input_dim))

    def save(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        elif len(os.listdir(self.model_dir)) != 0:
            fs = os.listdir(self.model_dir)
            for f in fs:
                os.remove(self.model_dir + f)
        save_path = self.saver.save(self.sess, self.model_dir + 'CRFasRNN.model', global_step=self.counter)
        print('MODEL RESTORED IN: ' + save_path)

    def load(self):
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, self.model_dir + ckpt_name)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


if __name__ == '__main__':
    reader = Reader()
    crf_rnn = CRF_RNN(input_shape=[512,512,1], batch_size=1, input_dim=262144, learning_rate=2e-4, pre_train=True)
    # crf_rnn.train(reader, loop=8000)

    for _ in range(30):
        xs, ys = reader.next_batch(1)
        pred = crf_rnn.predict(xs, as_list=False)
        print('--Precision: %g' % get_accuracy(pred, ys))

        pred = crf_rnn.predict(xs, as_list=True)
        for img in pred:
            plt.figure()
            plt.imshow(img, cmap='gray')
            plt.show()
        plt.close()

    print('\nFinish!!!')