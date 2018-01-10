'''
@Author: YukiRain
@Data: 2017.8.28

该文件主要负责将与界面无关的耗时计算分离到线程中，防止主界面的进程因为饥饿而被系统kill掉
该文件中所有的类应当继承QTCore.QThread类，并在其run函数中对耗时计算的过程进行进一步封装

@Date: 2017.10.28
集成了决策树的代码

@Date: 2017.13.21
集成了另一个版本的多尺度网络，未来可能会在这个网络后面加CRF
'''
from PyQt4 import QtCore
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.externals import joblib
from PIL import Image
import os, cv2

from dcm_read import dcm_reader
from levelset2D import LevelSet
from MLDT.Haar import haarReader, model_path
from segmentation import FCNet
from network import CRF_RNN

image_path = 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\1\\masks\\'

class levelsetThread(QtCore.QThread):
    # This signal controls the value of the ProgressBar on GUI
    signal = QtCore.pyqtSignal(int, name='signal')
    # And this one prints log on GUI
    send_msg = QtCore.pyqtSignal(str, name='send_msg')

    def __init__(self, path, parent=None, is_run=True):
        super(levelsetThread, self).__init__(parent=parent)
        # Working Directory
        self.path = path
        # The output
        self.data = None
        self.is_run =  is_run

    @staticmethod
    def process(img):
        img[380: , :] = 0
        return img

    def run(self):
        # Network for Segmentation
        self.send_msg.emit('FCNN Initializing...')
        time_1 = datetime.now()

        if self.is_run:
            self.fcnn = FCNet(learning_rate=1e-4)
            self.fcnn.load(path='E:\\Python\\surgeryGuidingProject\\deep_learning\\network\\')
            time_2 = datetime.now() - time_1
            self.send_msg.emit('FCNN Initialize Finished.\nTime Using:' + str(time_2))

        reader = dcm_reader(path=self.path)
        # arr = reader.getPixelArray()
        arr = self.process(reader.getPixelArray())
        slices, rows, cols = arr.shape
        data_slices = list()

        for i in range(slices):
            flatten = arr[i, :, :].reshape((1, rows*cols)).astype(np.float32)
            pred = self.fcnn.predict(flatten)
            pred = np.array(list(map(list, zip(*pred[::-1]))))
            pred = pred.transpose()
            # pred = np.array(list(map(list, zip(*pred[::-1]))))

            # levelset = LevelSet()
            # levelset.init_phi(pred)
            # levelset.evolution(iter_num=10, showed=False)
            # mask = levelset.get_contour(dtype=np.uint8)
            # data_slices.append(mask[None, :, :])

            Image.fromarray(pred).convert('RGB').save(image_path + str(i) + '.png')

            data_slices.append(pred[None, :, :])
            self.signal.emit(int(i * 100 / slices))

        self.data = np.concatenate(data_slices, axis=0)
        time_3 = datetime.now() - time_1
        self.send_msg.emit('Thread Finished.\nTime Cost: ' + str(time_3))


class decisionTreeThread(QtCore.QThread):
    # This signal controls the value of the ProgressBar on GUI
    signal = QtCore.pyqtSignal(int, name='signal')
    # And this one prints log on GUI
    send_msg = QtCore.pyqtSignal(str, name='send_msg')

    def __init__(self, path, parent=None):
        super(decisionTreeThread, self).__init__(parent=parent)
        self.path = path
        self.data = None

    def run(self):
        self.send_msg.emit('Decision Tree Initializing...')
        time_1 = datetime.now()

        self.dt = joblib.load(model_path + 'model.m')
        self.haar = haarReader()
        self.haar.setRandomWindowsFromFile(model_path + 'rands.npy')

        time_2 = datetime.now() - time_1
        self.send_msg.emit('Decision Tree Initialize Finished.\nTime Using:' + str(time_2))

        self.reader = dcm_reader(path=self.path)
        arr = self.reader.getPixelArray()
        slices, rows, cols = arr.shape
        data_slices = list()
        for i in range(slices):
            flatten = self.haar.getFeature(arr[i, :, :])
            pred = self.dt.predict(flatten).reshape((512, 512))

            data_slices.append(pred[None, :, :])
            self.signal.emit(int(i * 100 / slices))

        self.data = np.concatenate(data_slices, axis=0)
        time_3 = datetime.now() - time_1
        self.send_msg.emit('Thread Finished.\nTime Cost: ' + str(time_3))


class crfThread(QtCore.QThread):
    # This signal controls the value of the ProgressBar on GUI
    signal = QtCore.pyqtSignal(int, name='signal')
    # And this one prints log on GUI
    send_msg = QtCore.pyqtSignal(str, name='send_msg')

    def __init__(self, path, parent=None, is_run=False):
        super(crfThread, self).__init__(parent=parent)
        self.path = path
        self.data = None
        self.is_run = is_run

    @staticmethod
    def process(img):
        img = img.astype(np.float32)
        img[380: , :] = 0.0
        img[img < 0] = 0.0
        img /= 2048.0
        return img

    def run(self):
        # Network for Segmentation
        self.send_msg.emit('FCNN_V2 Initializing...')
        time_1 = datetime.now()

        if not self.is_run:
            self.fcnn = CRF_RNN(input_shape=[512,512,1], batch_size=1, input_dim=262144, pre_train=True)
            time_2 = datetime.now() - time_1
            self.send_msg.emit('FCNN_V2 Initialize Finished.\nTime Using:' + str(time_2))

        reader = dcm_reader(path=self.path)
        arr = self.process(reader.getPixelArray())
        slices, rows, cols = arr.shape
        data_slices = list()

        for i in range(slices):
            flatten = arr[i, :, :].reshape((1, rows*cols)).astype(np.float32)
            pred = self.fcnn.predict(flatten, as_list=True)[0]
            pred = np.array(list(map(list, zip(*pred[::-1]))))
            pred = pred.transpose() * 255.0

            Image.fromarray(pred).convert('RGB').save(image_path + str(i) + '.png')

            data_slices.append(pred[None, :, :])
            self.signal.emit(int(i * 100 / slices))

        self.data = np.concatenate(data_slices, axis=0)
        time_3 = datetime.now() - time_1
        self.send_msg.emit('Thread Finished.\nTime Cost: ' + str(time_3))


if __name__ == '__main__':
    thread = crfThread(path=None)
    for i in range(1, 11):
        if i == 4:
            continue

        treep = 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\%d\\3\\' % i
        image_path = 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\%d\\masks\\' % i
        thread.path = treep
        thread.run()
        print(i, 'next')
        thread.is_run = True