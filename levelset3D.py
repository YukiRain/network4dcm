# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image
from skimage import measure
import matplotlib.pyplot as plt
import tensorflow as tf

import levelset2D
from dcm_read import dcm_reader

sobel_kernel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]]).astype(np.float32)

class LevelSet(levelset2D.LevelSet):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, mu=0.001 * 255 * 255, reg_mu=1.0, time_step=0.1, epsilon=1.0):
        super(LevelSet, self).__init__(lambda_1=lambda_1,
                                       lambda_2=lambda_2,
                                       mu=mu,
                                       reg_mu=reg_mu,
                                       time_step=time_step,
                                       epsilon=epsilon)

        sobel_x = tf.constant(sobel_kernel, tf.float32)
        self.sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
        self.sobel_y_filter = tf.transpose(self.sobel_x_filter, perm=[1, 0, 2, 3])
        self.sobel_z_filter = tf.transpose(self.sobel_x_filter, perm=[1, 0, 2, 3])

    def init_phi(self, src, rect=None):
        self.src = src.astype(np.float32)
        self.src_op = tf.Variable(src, dtype=tf.float32)
        self.slices, self.rows, self.cols = src.shape
        self.area = self.slices * self.cols * self.cols

        if rect is None:
            rect = (30, self.slices-30, 30, self.rows-30, 30, self.cols-30)
        self._phi = np.ones(src.shape).astype(np.float32) * -2.0
        self._phi[rect[0]: rect[1], rect[2]: rect[3], rect[4]: rect[5]] = 2.0
        self._phi[rect[0], rect[2]: rect[3], rect[4]: rect[5]] = 0
        self._phi[rect[1], rect[2]: rect[3], rect[4]: rect[5]] = 0
        self._phi[rect[0]: rect[1], rect[2], rect[4]: rect[5]] = 0
        self._phi[rect[0]: rect[1], rect[3], rect[4]: rect[5]] = 0
        self._phi[rect[0]: rect[1], rect[2]: rect[3], rect[4]] = 0
        self._phi[rect[0]: rect[1], rect[2]: rect[3], rect[5]] = 0
        self.phi_op = tf.placeholder(tf.float32, [self.slices, self.rows, self.cols])

        k1 = self.epsilon / 3.14159265358979323846
        k2 = self.epsilon ** 2
        k3 = 2 / 3.14159265358979323846
        self.Dirac_op = k1 / (k2 + tf.pow(self.phi_op, 2.0))
        self.Heaviside_op = 0.5 * (k3 * tf.atan(self.phi_op / self.epsilon) + 1)
        self.curv_op = self._curvature(self.phi_op)
        self.fg_val_op, self.bg_val_op = self.binary_fit()
        self.penalizer_op = self._second_derivative(self.phi_op)

        self.length_term_op = self.mu * tf.multiply(self.Dirac_op, self.curv_op)
        self.penalize_term_op = self.reg_mu * (self.penalizer_op - self.curv_op)
        self.area_term_op = tf.multiply(self.Dirac_op, tf.subtract(
            self.lambda_2 * tf.pow(self.src_op - self.bg_val_op, 2.0),
            self.lambda_1 * tf.pow(self.src_op - self.fg_val_op, 2.0))
        )
        self.evolution_op = self.phi_op + self.time_step*(self.length_term_op + self.penalize_term_op + self.area_term_op)

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def _sobel(self, input_op):
        expand_op = tf.expand_dims(input_op, axis=3)
        transposed = tf.transpose(expand_op, perm=[1, 0, 2, 3])
        filtered_x = tf.nn.conv2d(expand_op, self.sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
        filtered_y = tf.nn.conv2d(expand_op, self.sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
        filtered_z = tf.nn.conv2d(transposed, self.sobel_z_filter, strides=[1, 1, 1, 1], padding='SAME')
        transpose_z = tf.transpose(filtered_z, perm=[1, 0, 2, 3])
        return filtered_x, filtered_y, transpose_z

    def _curvature(self, m_input):
        img_dx, img_dy, img_dz = self._sobel(m_input)
        normalizer = tf.sqrt(tf.pow(img_dx, 2) + tf.pow(img_dy, 2) + tf.pow(img_dz, 2) + 1e-10)
        img_dx = tf.mod(img_dx, normalizer)
        img_dy = tf.mod(img_dy, normalizer)
        img_dz = tf.mod(img_dz, normalizer)

        dxdx = tf.nn.conv2d(img_dx, self.sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
        dydy = tf.nn.conv2d(img_dy, self.sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
        dzdz = tf.nn.conv2d(img_dz, self.sobel_z_filter, strides=[1, 1, 1, 1], padding='SAME')
        output = dxdx + dydy + dzdz
        return tf.reshape(output, shape=(self.slices, self.rows, self.cols))

    def _second_derivative(self, m_input):
        img_dx, img_dy, img_dz = self._sobel(m_input)
        dxdx = tf.nn.conv2d(img_dx, self.sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
        dydy = tf.nn.conv2d(img_dy, self.sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
        dzdz = tf.nn.conv2d(img_dz, self.sobel_z_filter, strides=[1, 1, 1, 1], padding='SAME')
        output =  tf.sqrt(tf.pow(dxdx, 2) + tf.pow(dydy, 2) + tf.pow(dzdz, 2))
        return tf.reshape(output, shape=(self.slices, self.rows, self.cols))

    def binary_fit(self):
        fg_val = tf.reduce_mean(tf.multiply(self.Heaviside_op, self.src_op))
        bg_val = tf.reduce_mean(tf.multiply(1 - self.Heaviside_op, self.src_op))
        return fg_val, bg_val

    def evolution_once(self):
        self._phi = self.sess.run(self.evolution_op, feed_dict={self.phi_op: self._phi})

    def show_contour(self):
        rgb_src = np.array(Image.fromarray(self.src[50, :, :]).convert('RGB'))
        mask = np.zeros_like(self._phi[50, :, :]).astype(np.int32)
        mask[self._phi[50, :, :] > 0] = 255
        contours = measure.find_contours(mask, level=0.5)
        plt.figure()
        plt.imshow(rgb_src)
        for ct in contours:
            plt.plot(ct[:, 1], ct[:, 0], linewidth=2, color='c')
        plt.show()
        # for ct in contours:
        #     rgb_src[ct[:, 0], ct[:, 1]] = (0, 255, 255)
        # Image.fromarray(rgb_src).save('6.png')
        print('Evolution result saved!')


if __name__ == '__main__':
    reader = dcm_reader(path='E:\\C++\\Projects\\surgeryGuidingProject_copy\\8848\\new\DICOM\\20170510\\08290000\\6\\')
    arr = reader.getPixelArray()
    arr = reader.dicom_normalize(arr)
    levelset3 = LevelSet()
    levelset3.init_phi(arr)
    levelset3.evolution()