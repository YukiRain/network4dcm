'''
----------------------------------------------------------------------------
Origin source code of 3D CV-levelset
该程序慢到让人怀疑人生，计划用Tensorflow重写一遍

@Author: YukiRain
@Date: 2017.8.23

----------------------------------------------------------------------------
'''

import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure

import levelset2D
from dcm_read import dcm_reader

class LevelSet(levelset2D.LevelSet):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, mu=0.001 * 255 * 255, reg_mu=1.0, time_step=0.1, epsilon=1.0):
        super(LevelSet, self).__init__(lambda_1=lambda_1,
                                       lambda_2=lambda_2,
                                       mu=mu,
                                       reg_mu=reg_mu,
                                       time_step=time_step,
                                       epsilon=epsilon)
        self._curv = None

    def init_phi(self, src, rect=None):
        self.src = src.astype(np.float32)
        self.slices, self.rows, self.cols = src.shape
        self.area = self.slices * self.cols * self.cols
        self._Dirac = np.zeros_like(self.src).astype(np.float32)
        self._Heaviside = np.zeros_like(self.src).astype(np.float32)

        if rect is None:
            rect = (30, self.slices-30, 30, self.rows-30, 30, self.cols-30)
        self._phi = np.ones(self.src.shape).astype(np.float32) * -2.0
        self._phi[rect[0]: rect[1], rect[2]: rect[3], rect[4]: rect[5]] = 2
        self._phi[rect[0], rect[2]: rect[3], rect[4]: rect[5]] = 0
        self._phi[rect[1], rect[2]: rect[3], rect[4]: rect[5]] = 0
        self._phi[rect[0]: rect[1], rect[2], rect[4]: rect[5]] = 0
        self._phi[rect[0]: rect[1], rect[3], rect[4]: rect[5]] = 0
        self._phi[rect[0]: rect[1], rect[2]: rect[3], rect[4]] = 0
        self._phi[rect[0]: rect[1], rect[2]: rect[3], rect[5]] = 0

    def _curvature(self, m_input):
        img_dx = np.zeros(m_input.shape).astype(np.float32)
        img_dy = np.zeros(m_input.shape).astype(np.float32)
        img_dz = np.zeros(m_input.shape).astype(np.float32)
        filters.sobel(m_input, 2, img_dx)
        filters.sobel(m_input, 1, img_dy)
        filters.sobel(m_input, 0, img_dz)

        normalizer = np.sqrt(img_dx**2 + img_dy**2 + img_dz**2 + 1e-10)
        img_dx = img_dx / normalizer
        img_dy = img_dy / normalizer
        img_dz = img_dz / normalizer

        dxdx = filters.sobel(img_dx, 2)
        dydy = filters.sobel(img_dy, 1)
        dzdz = filters.sobel(img_dz, 0)
        return dxdx + dydy + dzdz

    def _second_derivative(self, m_input):
        img_dx = filters.sobel(m_input, 2)
        img_dy = filters.sobel(m_input, 1)
        img_dz = filters.sobel(m_input, 0)
        dxdx = filters.sobel(img_dx, 2)
        dydy = filters.sobel(img_dy, 1)
        dzdz = filters.sobel(img_dz, 0)
        return np.sqrt(dxdx**2 + dydy**2 + dzdz**2)

    def evolution_once(self):
        self._Dirac = self._Dirac_func(self._phi)
        self._curv = self._curvature(self._phi)
        fg_val, bg_val = self.binary_fit()
        penalizer = self._second_derivative(self._phi)

        length_term = self.mu * self._Dirac * self._curv
        penalize_term = self.reg_mu * (penalizer - self._curv)
        area_term = self._Dirac * (self.lambda_2 * ((self.src - bg_val)**2) - self.lambda_1 * ((self.src - fg_val)**2))
        self._phi = self._phi + self.time_step * (length_term + penalize_term + area_term)

    def show_contour(self):
        for i in range(self.slices):
            rgb_src = np.array(Image.fromarray(self.src[i, :, :]).convert('RGB'))
            mask = np.zeros_like(self._phi[i, :, :]).astype(np.int32)
            mask[self._phi[i, :, :] > 0] = 255
            contours = measure.find_contours(mask, level=0.5)
            plt.figure()
            plt.imshow(rgb_src)
            for ct in contours:
                plt.plot(ct[:, 1], ct[:, 0], linewidth=2, color='c')
            plt.show()


if __name__ == '__main__':
    reader = dcm_reader(path='E:\\C++\\Projects\\surgeryGuidingProject_copy\\8848\\new\DICOM\\20170510\\08290000\\')
    arr = reader.getPixelArray()
    arr = reader.dicom_normalize(arr)
    levelset3 = LevelSet()
    levelset3.init_phi(arr)
    levelset3.evolution()