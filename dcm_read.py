import dicom
import numpy as np
import matplotlib.pyplot as plt
import os, shutil, cv2
from skimage import measure


class dcm_reader(object):
    def __init__(self, path):
        # 传dicom文件所在目录进去，数据按照文件名索引slice存放在字典中
        fileNames = os.listdir(path)
        fileNames = list(map(lambda x: path + x, fileNames))
        fileNames = list(filter(lambda x: '.dcm' in x, fileNames))
        self._path = path
        self.data = dict()
        for name in fileNames:
            self.data[name] = dicom.read_file(name)

    def splitToFolders(self):
        '''
        :把相同序列号的dcm文件放到同一个文件夹下
        注意：原本文件夹路径下不要有除了dcm的任何别的东西，否则会被删掉
        '''
        ret = list()
        folders = list(filter(lambda x: '.dcm' not in x, os.listdir(self._path)))
        for f in folders:
            shutil.rmtree(self._path + f)
        for key, val in self.data.items():
            if str(val.SeriesNumber) not in os.listdir(self._path):
                os.makedirs(self._path + str(val.SeriesNumber))
                print('New Series: ' + str(val.SeriesNumber))
                if len(val.pixel_array) == (512, 512):
                    ret.append(val.pixel_array)
            shutil.copy(key, self._path + str(val.SeriesNumber) + '\\' + key.split('\\')[-1])
        print('Finished!!')
        return ret

    def getPixelArray(self, slice = -1):
        '''
        :return: dicom序列拼成的三维矩阵
        '''
        arr_list = list(self.data.values())
        arr_list.sort(key=lambda x: int(x.data_element('InstanceNumber').value))
        if slice == -1:
            data = [item.pixel_array[None, :, :] for item in arr_list]
            data = list(filter(lambda x: x.shape==(1,512,512), data))
            return np.concatenate(data, axis=0)
        else:
            ls = list(self.data.values())
            return ls[slice].pixel_array

    def getSliceThcikness(self):
        # 返回z轴上每片的厚度
        arr_list = list(self.data.values())
        return arr_list[0].data_element('SliceThickness').value

    def list(self, slice=-1):
        # 把某一片的数据打印在屏幕上，如果slice等于-1，打印所有片的数据
        if slice != -1:
            items = list(self.data.values())
            for it in items:
                for i in it:
                    print(i)
                print('--------------------------------------')
        else:
            item = list(self.data.values())[slice]
            for it in item:
                print(it)

    def getPatientInfo(self):
        ls = list(self.data.values())
        return ls[0].data_element('PatientName').value,\
               ls[0].data_element('PatientSex').value,\
               ls[0].data_element('PatientID').value

    def splitSeries(self):
        output = dict()
        for key, val in self.data.items():
            seriesNumber = int(val.SeriesNumber)
            if seriesNumber not in output.keys():
                output[seriesNumber] = list()
            output[seriesNumber].append(key)
        for val in self.data.values():
            val.sort(key=lambda x: int(x.data_element('InstanceNumber').value))
        return output

    @staticmethod
    def dicom_normalize(img):
        # 将CT值转成0-255的灰度空间，该转换方案仍有待改进
        img[img < 0] = 0
        maxVal = np.max(img)
        minVal = np.min(img)
        return (img.astype(np.float32) - minVal) / (maxVal - minVal) * 255.0

    def save(self, path):
        arr = self.dicom_normalize(self.getPixelArray())
        for i in range(arr.shape[0]):
            cv2.imwrite(path + str(i) + '.png', arr[i, :, :])
        print('Saved to ' + path)


def draw_contours(arr_list, label_list, save=False, show=True, path=None):
    cnt = 0
    for x, y in zip(arr_list, label_list):
        y = cv2.flip(y, 0)
        contours = measure.find_contours(y[:, :, 0], level=150.0)
        plt.figure()
        plt.imshow(x)
        for ct in contours:
            plt.plot(ct[:, 1], ct[:, 0], linewidth=1, color='c')
        if show:
            plt.show()
        if save:
            plt.savefig(path + str(cnt) + '.png')
        cnt += 1
        plt.close()

def convert_index(path):
    fsn = os.listdir(path)
    for fn in fsn:
        os.rename(path + fn, path + '_' + fn)
    idx = sorted(fsn, key=lambda x: int(x.split('.')[0]))
    fsn = os.listdir(path)
    for i, j in zip(fsn, idx):
        os.rename(path + i, path + j)
    print('--path: %s --number: %d' % (path, len(fsn)))


if __name__ == '__main__':
    for i in range(1, 11):
        if i == 4:
            continue
        treep = 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\%d\\3\\' % i
        image_path = 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\%d\\orgs\\' % i
        label_path = 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\%d\\masks\\' % i
        edgep = 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\raw_data\\%d\\edges\\' % i
        if not os.path.exists(edgep):
            os.mkdir(edgep)

        xfs = os.listdir(image_path); yfs = os.listdir(label_path)
        xfs = sorted(xfs, key=lambda x: int(x.split('.')[0]))
        yfs = sorted(yfs, key=lambda x: int(x.split('.')[0]))
        xfiles = list(map(lambda x: cv2.imread(image_path + x), xfs))
        yfiles = list(map(lambda x: cv2.imread(label_path + x), yfs))
        for yf, y in zip(yfs, yfiles):
            y = cv2.flip(y, 0)
            cv2.imwrite(label_path + yf, y)
        draw_contours(xfiles, yfiles, show=False, save=True, path=edgep)
        print('--NEXT: %d' % i)