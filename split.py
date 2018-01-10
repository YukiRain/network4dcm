from dcm_read import dcm_reader
import os, sys, shutil
import PyQt4.uic as pyuic
from PyQt4 import QtCore, QtGui
from PIL import Image
import numpy as np

new_path = 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\8848\\new\DICOM\\20170510\\08290000\\'

class splitWindow(QtGui.QMainWindow):
    def __init__(self, parent=None, path=new_path):
        QtGui.QMainWindow.__init__(self, parent)
        self.to_del = list()
        self._choice = None
        self.OK = False
        self.ui = pyuic.loadUi('split.ui', self)
        self._initialize(path)
        self._get_info()
        self.ui.buttonOK.clicked.connect(self._ok_option)

    def _initialize(self, path):
        self.reader = dcm_reader(path)
        pixLabels = [self.ui.png1, self.ui.png2, self.ui.png3]
        chart = dict()
        for key, val in self.reader.data.items():
            to_path = '\\'.join(key.split('\\')[: -1]) + '\\' + str(val.SeriesNumber) + '\\'
            if str(val.SeriesNumber) not in chart:
                # 如果已经有上次使用过的文件夹还存在，没有被删除的话，就先删掉
                if str(val.SeriesNumber) in os.listdir('\\'.join(key.split('\\')[: -1]) + '\\'):
                    shutil.rmtree(to_path)
                # 新建文件夹
                os.makedirs(to_path)
                self.to_del.append(to_path)
                # 保存图片到本地
                try:
                    img_to_save = self._dicom_normalize(val.pixel_array)
                    Image.fromarray(img_to_save).convert('RGB').save(str(val.SeriesNumber) + '.png')
                    chart[str(val.SeriesNumber)] = str(val.SeriesNumber) + '.png'
                except:
                    print('Something wrong')
                    continue
            shutil.copy(key, to_path + key.split('\\')[-1])
        # 把上面保存在本地的图片文件渲染到界面上
        for index, pic in enumerate(chart.values()):
            if index >= 3:
                break
            png = QtGui.QPixmap(pic)
            png = png.scaled(175, 175,
                             QtCore.Qt.IgnoreAspectRatio,
                             QtCore.Qt.SmoothTransformation)
            pixLabels[index].setPixmap(png)
        self.ui.statusBar.showMessage('Initialize Finished')

    def _get_info(self):
        name, sex, id = self.reader.getPatientInfo()
        self.ui.labelName.setText('Patient Name: ' + str(name))
        self.ui.labelSex.setText('Patient Sex: ' + str(sex))
        self.ui.labelID.setText('Patient ID: ' + str(id))
        self.ui.labelPath.setText('\n'.join(self.to_del))
        textWidgets = [self.ui.text1, self.ui.text2, self.ui.text3]
        for folder, tw in zip(self.to_del, textWidgets):
            num = len(os.listdir(folder))
            tw.setText('Number of Slices: ' + str(num))
        self.ui.statusBar.showMessage('Initialize Patient information Finished')

    def _ok_option(self):
        if self._choice == None:
            QtGui.QMessageBox.warning(self, 'ERROR', 'One Series must be chosen!', QtGui.QMessageBox.Ok)
            return None
        seriesPaths = self.ui.labelPath.text().split('\n')
        try:
            del seriesPaths[self._choice]
        except:
            QtGui.QMessageBox.warning(self, 'ERROR', 'A blank series of DICOM files are chosen', QtGui.QMessageBox.Ok)
            return None
        # 只留下被选择的那个序列对应的文件夹，其他都直接删掉，最后关闭本窗口
        for sp in seriesPaths:
            shutil.rmtree(sp)
        self.OK = True
        self.close()

    def _dicom_normalize(self, img):
        img[img < 0] = 0
        maxVal = np.max(img)
        minVal = np.min(img)
        return (img.astype(np.float32) - minVal) / (maxVal - minVal) * 255.0

    def mousePressEvent(self, event):
        x = event.x(); y = event.y()
        self.ui.statusBar.showMessage('X: %d; Y: %d' % (x, y))
        if y > 215 and y < 400:
            if x > 35 and x < 220:
                self._choice = 0
                self.update()
            elif x >= 220 and x <= 400:
                self._choice = 1
                self.update()
            elif x > 400 and x < 580:
                self._choice = 2
                self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setPen(QtGui.QPen(QtCore.Qt.gray, QtCore.Qt.DashDotLine))
        painter.setBrush(QtGui.QBrush(QtCore.Qt.yellow, QtCore.Qt.SolidPattern))
        try:
            x_begin = 35 + self._choice * 185
            painter.drawRect(x_begin, 205, 185, 200)
        finally:
            painter.end()
            return None


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = splitWindow()
    window.show()
    # for it in window.to_del:
    #     shutil.rmtree(it)
    sys.exit(app.exec_())