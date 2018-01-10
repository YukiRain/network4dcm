from PIL import Image
import numpy as np
import os
import dicom
import matplotlib.pyplot as plt
from functools import reduce

src_path = 'E:\\C++\\Projects\\TrainingData\\Final!!!\\data\\'
label_path = 'E:\\C++\\Projects\\TrainingData\\Final!!!\\label\\'

class Reader(object):
    def __init__(self, x_path=src_path, y_path=label_path):
        self.x_data = list()
        self.y_data = list()
        self.x_files = self._run_path(x_path)
        self.y_files = self._run_path(y_path)
        self.train, self.labels = self._read()
        self.hashing = np.zeros((len(self.train))).astype(np.uint8)

    @staticmethod
    def _run_path(path):
        fs = os.listdir(path)
        fs = list(map(lambda x: path + x, fs))
        return fs

    def _read(self, img_size=512, x_norm=2048.0, y_norm=255.0):
        for x_name, y_name in zip(self.x_files, self.y_files):
            x_file = dicom.read_file(x_name)
            x_arr = np.array(x_file.pixel_array, dtype=np.float32).reshape((1, img_size * img_size))
            self.x_data.append(x_arr)
            y_img = Image.open(y_name).convert('L')
            y_arr = np.array(y_img, dtype=np.float32).reshape((1, img_size*img_size))
            self.y_data.append(y_arr)
        x_array = np.concatenate(tuple(self.x_data), axis=0)
        y_array = np.concatenate(tuple(self.y_data), axis=0)
        print('Reader Initialization: Finished')
        print('Training Data Size: %d \nTest Data Size: %d' % (len(self.x_data), len(self.y_data)))
        x_array[x_array < 0.0] = 0.0
        x_array /= x_norm
        y_array /= y_norm
        return x_array, y_array

    def next_batch(self, num):
        index = np.random.randint(low=0, high=len(self.x_data), size=(1, num))
        x_batch = self.train[index]
        y_batch = self.labels[index]
        self.hashing[index] = 1
        # return x_batch[0] + np.random.normal(scale=120, size=x_batch[0].shape),\
        #        y_batch[0] + np.random.normal(scale=120, size=x_batch[0].shape)
        return x_batch[0], y_batch[0]

    def ordered_batch(self):
        output = list()
        for i in range(0, len(self.x_data)):
            index = np.array([[i]])
            x_batch = self.train[index]
            output.append(x_batch[0])
        return output

    # 训练时注意控制样本数量与训练迭代采样次数成正比
    def get_test_data(self):
        idx = (self.hashing == 0)
        test_x = self.train[idx]
        test_y = self.labels[idx]
        print('%d samples remained for test!' % test_x.shape[0])
        return test_x, test_y


if __name__ == '__main__':
    reader = Reader()
    for i in range(20):
        _ = reader.next_batch(5)

    test_x, test_y = reader.get_test_data()

    for i in range(test_x.shape[0]):
        arr = test_x[i].reshape((512, 512))
        plt.figure()
        plt.subplot(121)
        plt.imshow(arr, cmap='gray')
        plt.subplot(122)
        plt.imshow(test_y[i].reshape((512, 512)), cmap='gray')
        plt.show()

    print('Done!!')