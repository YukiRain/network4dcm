# -*- coding:utf-8 -*-
import numpy as np
from scipy.ndimage import filters
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure
from datetime import datetime

from dcm_read import dcm_reader

class LevelSet(object):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, mu=0.001*255*255, reg_mu=1.0, time_step=0.1, epsilon=1.0):
        '''
        :param lambda_1: 全局项系数(inside)
        :param lambda_2: 全局项系数(outside)
        :param mu: 长度约束项系数系数
        :param reg_mu: 李纯明提出的能量惩罚项系数
        :param time_step: 演化步长
        :param epsilon: Heaviside函数与Dirac函数的规则化参数
        '''
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.mu = mu
        self.reg_mu = reg_mu
        self.time_step = time_step
        self.epsilon = epsilon

    def init_phi(self, src, rect=None):
        '''
        :param src: src image
        :param rect: format like(x1, x2, y1, y2)
        _phi: 水平集phi函数
        _Heaviside: 海氏函数处理后水平集
        _Dirac： Dirac函数处理后水平集
        _mK: 惩罚项卷积核，用一步卷积完成求phi的二阶梯度这一步复杂操作
            貌似是Laplace算子的一种？？？
        '''
        self.src = src.astype(np.float32)
        self.rows, self.cols = src.shape
        self.area = self.rows * self.cols
        self._Dirac = np.zeros_like(self.src).astype(np.float32)
        self._Heaviside = np.zeros_like(self.src).astype(np.float32)
        self._mK = np.array([[0.5, 1.0, 0.5],
                             [1.0, -6.0, 1.0],
                             [0.5, 1.0, 0.5]])
        self._curv = None
        # 初始化phi，矩形的框上面为0，外面为-2，里面为2
        if rect is None:
            rect = (30, self.rows-30, 30, self.cols-30)
        self._phi = np.ones(self.src.shape).astype(np.float32) * -2.0
        self._phi[rect[0]: rect[1], rect[2]: rect[3]] = 2
        self._phi[rect[0], rect[2]: rect[3]] = 0
        self._phi[rect[1], rect[2]: rect[3]] = 0
        self._phi[rect[0]: rect[1], rect[2]] = 0
        self._phi[rect[0]: rect[1], rect[3]] = 0

    def _Dirac_func(self, m_input):
        '''
        :param m_input: self._phi
        :return: self._Dirac
        '''
        k1 = self.epsilon / 3.14159265358979323846
        k2 = self.epsilon ** 2
        return k1 / (k2 + m_input**2)

    def _Heaviside_func(self, m_input):
        '''
        :param m_input: self._phi
        :return: self._Heaviside
        '''
        k3 = 2 / 3.14159265358979323846
        return 0.5 * (1 + k3 * np.arctan(m_input/self.epsilon))

    def _curvature(self, m_input):
        '''
        :param m_input: self._phi
        :return: self._curv
        '''
        # 一阶导
        img_dx = np.zeros(m_input.shape).astype(np.float32)
        img_dy = np.zeros(m_input.shape).astype(np.float32)
        filters.sobel(m_input, 1, img_dx)
        filters.sobel(m_input, 0, img_dy)
        # 梯度归一化
        normalizer = np.sqrt(img_dx**2 + img_dy**2 + 1e-10)
        img_dx = img_dx / normalizer
        img_dy = img_dy / normalizer
        # 曲率等于归一化梯度的散度
        dxdx = filters.sobel(img_dx, 1)
        dydy = filters.sobel(img_dy, 0)
        return dxdx + dydy

    def binary_fit(self):
        '''
        :return: 演化曲线内部和外部的平均灰度
        '''
        self._Heaviside = self._Heaviside_func(self._phi)
        fg_area = self._Heaviside.sum()
        bg_kernel = 1 - self._Heaviside
        fg_sum = (self._Heaviside * self.src).sum()
        bg_sum = (bg_kernel * self.src).sum()
        fg_value = fg_sum / (fg_area + 1e-10)
        bg_value = bg_sum / (self.area - fg_area + 1e-10)
        return fg_value, bg_value

    def evolution_once(self):
        self._Dirac = self._Dirac_func(self._phi)
        self._curv = self._curvature(self._phi)
        fg_val, bg_val = self.binary_fit()
        penalizer = cv2.filter2D(self._phi, cv2.CV_32F, self._mK, (1, 1))

        length_term = self.mu * self._Dirac * self._curv
        penalize_term = self.reg_mu * (penalizer - self._curv)
        area_term = self._Dirac * (self.lambda_2 * ((self.src - bg_val)**2) - self.lambda_1 * ((self.src - fg_val)**2))
        self._phi = self._phi + self.time_step * (length_term + penalize_term + area_term)

    def evolution(self, iter_num=10, showed=True):
        for iter in range(iter_num):
            # print(datetime.now(), ': Evolution iterate: ' + str(iter))
            self.evolution_once()
        if showed:
            self.show_contour()

    def show_contour(self):
        rgb_src = np.array(Image.fromarray(self.src).convert('RGB'))
        mask = np.zeros_like(self._phi).astype(np.int32)
        mask[self._phi > 0] = 255
        contours = measure.find_contours(mask, level=0.5)
        plt.figure()
        plt.imshow(rgb_src)
        for ct in contours:
            plt.plot(ct[:, 1], ct[:, 0], linewidth=2, color='c')
        plt.show()

    def get_contour(self, contour_val=255, dtype=np.uint8):
        mask = np.zeros_like(self._phi).astype(dtype)
        mask[self._phi > 0] = contour_val
        # contours = measure.find_contours(mask, level=0.5)
        # output = np.zeros_like(self._phi).astype(dtype)
        # for item in contours:
        #     item_copy = item.astype(np.int)
        #     output[item_copy[:, 0], item_copy[:, 1]] = contour_val
        return mask


if __name__ == '__main__':
    # img = Image.open('E:\\test.jpg').convert('L')
    # levelset = LevelSet()
    # levelset.init_phi(np.array(img))
    # levelset.evolution(showed=True)

    reader = dcm_reader(path='E:\\C++\\Projects\\surgeryGuidingProject_copy\\8848\\new\\DICOM\\20170510\\08290000\\6\\')
    arr = reader.getPixelArray()
    arr = reader.dicom_normalize(arr)
    for i in range(arr.shape[0]):
        slice = arr[i, :, :]
        max_val = slice.max()
        # slice[slice < 0.75*max_val] = 0

        levelset = LevelSet()
        levelset.init_phi(slice)

        levelset.evolution()