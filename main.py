import sys, os
import vtk
from PyQt4 import QtCore, QtGui
import PyQt4.uic as pyuic
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support
from datetime import datetime

from split import splitWindow
from multi_thread import levelsetThread, decisionTreeThread, crfThread

new_path = 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\new_data\\series2\\7\\'

class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.ui = pyuic.loadUi('noVTK.ui', self)
        self.setWindowState(QtCore.Qt.WindowMaximized)
        self._initUnusedVariable()
        self.ui.progressBar.setVisible(False)

        self.vtk3DWidget = QVTKRenderWindowInteractor(self.ui.frame)
        self.xy_plane = QVTKRenderWindowInteractor(self.ui.frame_xy)
        self.yz_plane = QVTKRenderWindowInteractor(self.ui.frame_yz)
        self.xz_plane = QVTKRenderWindowInteractor(self.ui.frame_xz)
        self.innerLayout = QtGui.QVBoxLayout()
        self.ui.frame.setLayout(self.innerLayout)
        self.innerLayout.addWidget(self.vtk3DWidget)

        self.ui.slide1.valueChanged.connect(self.ui.spinBox.setValue)
        self.ui.slide2.valueChanged.connect(self.ui.spinBox1.setValue)
        self.ui.slide3.valueChanged.connect(self.ui.spinBox2.setValue)
        self.ui.spinBox.valueChanged.connect(self.ui.slide1.setValue)
        self.ui.spinBox1.valueChanged.connect(self.ui.slide2.setValue)
        self.ui.spinBox2.valueChanged.connect(self.ui.slide3.setValue)
        self.ui.mc_spin.setRange(0, 1000)
        self.ui.mc_slide.setMinimum(0)
        self.ui.mc_slide.setMaximum(1000)
        self.ui.mc_slide.valueChanged.connect(self.ui.mc_spin.setValue)
        self.ui.mc_slide.sliderReleased.connect(self.setMarchingCubesValue)
        self.ui.mc_spin.valueChanged.connect(self.ui.mc_slide.setValue)

        self.ui.actionOpen.triggered.connect(self._open_dcm)
        self.ui.actionRefresh.triggered.connect(self._initUnusedVariable)
        self.ui.actionSave.triggered.connect(self.export2stl)
        self.ui.action_save_mhd.triggered.connect(self.export2mhd)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.buttonMC.clicked.connect(self._MarchingCubes)
        self.ui.buttonLevelset.clicked.connect(self._levelset)
        self.ui.buttonML.clicked.connect(self._MachineLearning)
        self.ui.buttonCRF.clicked.connect(self._crf_as_rnn)
        self.ui.slide1.sliderReleased.connect(self.updateAxisX)
        self.ui.slide2.sliderReleased.connect(self.updateAxisY)
        self.ui.slide3.sliderReleased.connect(self.updateAxisZ)

    def _initUnusedVariable(self):
        self.reader = None
        self._path = None
        self.ren = None; self.ren1 = None; self.ren2 = None; self.ren3 = None
        self.iren = None; self.iren1 = None; self.iren2 = None; self.iren3 = None
        self.ui.buttonLevelset.setEnabled(False)
        self.ui.buttonMC.setEnabled(False)
        self.ui.buttonML.setEnabled(False)
        self.ui.buttonCRF.setEnabled(False)
        self.ui.logging.append('Variables Initialized')

    def _setSliderValue(self, dims):
        self.ui.slide1.setMinimum(0)
        self.ui.slide1.setMaximum(dims[0])
        self.ui.slide2.setMinimum(0)
        self.ui.slide2.setMaximum(dims[1])
        self.ui.slide3.setMinimum(0)
        self.ui.slide3.setMaximum(dims[2])
        self.ui.spinBox.setRange(0, dims[0])
        self.ui.spinBox1.setRange(0, dims[1])
        self.ui.spinBox2.setRange(0, dims[2])

    def _levelset_pipeline(self):
        self.ren = vtk.vtkRenderer()
        self.vtk3DWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtk3DWidget.GetRenderWindow().GetInteractor()

        self.ui.progressBar.setValue(0)
        slices, rows, cols = self.thread_levelset.data.shape
        self.ui.logging.append('Reconstructing...')

        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(self._path)
        self.reader.GlobalWarningDisplayOff()
        self.reader.Update()

        # The numpy_support of VTK only SHALLOW-COPIES the array, so keep the array survive!!!
        # Need to transpose x and z coordinate because VTK's data ordering is DIFFERENT
        self.flat_array = self.thread_levelset.data.transpose(2, 1, 0).flatten()
        self.dataArray = numpy_support.numpy_to_vtk(self.flat_array)
        self.ui.progressBar.setValue(20)

        self.imageData = vtk.vtkImageData()
        self.imageData.SetDimensions(slices, rows, cols)
        self.imageData.GetPointData().SetScalars(self.dataArray)
        # self.thickness
        self.ui.logging.append('MarchingCubes Update Finished')
        self.imageData.SetSpacing(self.thickness, 0.625, 0.625)
        self.imageData.SetOrigin(0, 0, 0)

        self.mc = vtk.vtkMarchingCubes()
        self.mc.SetInputData(self.imageData)
        self.mc.SetValue(0, 150)
        self.mc.Update()
        self.ui.progressBar.setValue(50)
        self.ui.logging.append('MarchingCubes Update Finished')

        self.tri = vtk.vtkTriangleFilter()
        self.tri.SetInputConnection(self.mc.GetOutputPort())
        self.tri.Update()

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.tri.GetOutputPort())
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.ren.AddActor(self.actor)
        self.ren.SetBackground(0.9, 0.9, 0.9)
        self.ren.ResetCamera()
        self.ui.progressBar.setValue(80)
        self.iren.Initialize()
        self.ui.logging.append('Reconstruction Finished')
        self.ui.progressBar.setVisible(False)

        self.seg_arr = numpy_support.vtk_to_numpy(self.reader.GetOutput().GetPointData().GetScalars())
        self.segmentDataArray = numpy_support.numpy_to_vtk(self.seg_arr.flatten())
        self.segImageData = vtk.vtkImageData()
        self.segImageData.SetDimensions(slices, rows, cols)
        self.segImageData.SetSpacing(self.thickness, 1, 1)
        self.segImageData.SetOrigin(0, 0, 0)

        self._threeViewPipeline(self.ui.spinBox.value(), self.ui.spinBox1.value(), self.ui.spinBox2.value())
        self.iren1.Initialize()
        self.iren2.Initialize()
        self.iren3.Initialize()

    def _construct_pipeline(self, path, mc=500, thresh=350):
        self.ui.logging.append('Reconstructing...')
        self.ren = vtk.vtkRenderer()
        # self.ren.SetLayer(1)
        # self.vtk3DWidget.GetRenderWindow().SetNumberOfLayers(2)
        self.vtk3DWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtk3DWidget.GetRenderWindow().GetInteractor()
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(path)
        self.reader.GlobalWarningDisplayOff()
        self.reader.Update()

        dims = self.reader.GetOutput().GetDimensions()
        self.readerImageCast = vtk.vtkImageCast()
        self.readerImageCast.SetInputConnection(self.reader.GetOutputPort())
        self.readerImageCast.SetOutputScalarTypeToFloat()
        self.readerImageCast.Update()
        # 把三视图的三个滑块的取值范围设定成三个维度的大小
        self._setSliderValue(dims)

        # self.thresh = vtk.vtkImageThreshold()
        # self.thresh.SetInputConnection(self.readerImageCast.GetOutputPort())
        # self.thresh.ThresholdByUpper(thresh)
        # self.thresh.SetOutValue(0)
        # self.thresh.Update()

        self.mc = vtk.vtkMarchingCubes()
        self.mc.SetInputConnection(self.readerImageCast.GetOutputPort())
        self.mc.SetValue(0, mc)
        self.mc.Update()

        self.tri = vtk.vtkTriangleFilter()
        self.tri.SetInputConnection(self.mc.GetOutputPort())
        self.tri.Update()

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.tri.GetOutputPort())
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.ren.AddActor(self.actor)
        self.ren.SetBackground(0.9, 0.9, 0.9)
        self.ren.ResetCamera()
        self.ui.logging.append('Reconstruction Finished')

    def _threeViewPipeline(self, x, y, z):
        self.ren1 = vtk.vtkRenderer()
        self.ren2 = vtk.vtkRenderer()
        self.ren3 = vtk.vtkRenderer()
        self.ren1.SetBackground(0.9, 0.9, 0.9)
        self.ren2.SetBackground(0.9, 0.9, 0.9)
        self.ren3.SetBackground(0.9, 0.9, 0.9)
        self.xy_plane.GetRenderWindow().AddRenderer(self.ren1)
        self.yz_plane.GetRenderWindow().AddRenderer(self.ren2)
        self.xz_plane.GetRenderWindow().AddRenderer(self.ren3)

        self.iren1 = self.xy_plane.GetRenderWindow().GetInteractor()
        self.iren2 = self.yz_plane.GetRenderWindow().GetInteractor()
        self.iren3 = self.xz_plane.GetRenderWindow().GetInteractor()

        self.planeWidgetX = vtk.vtkImagePlaneWidget()
        self.planeWidgetY = vtk.vtkImagePlaneWidget()
        self.planeWidgetZ = vtk.vtkImagePlaneWidget()
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)

        self.planeWidgetX.SetInteractor(self.iren1)
        self.planeWidgetX.SetPicker(self.picker)
        self.planeWidgetX.TextureInterpolateOff()
        self.planeWidgetX.RestrictPlaneToVolumeOn()
        self.planeWidgetX.SetResliceInterpolateToLinear()
        self.planeWidgetX.SetInputData(self.reader.GetOutput())
        self.planeWidgetX.SetPlaneOrientationToXAxes()
        self.planeWidgetX.SetSliceIndex(x)
        self.planeWidgetX.GetTexturePlaneProperty().SetOpacity(1)
        self.planeWidgetX.On()

        self.planeWidgetY.SetInteractor(self.iren2)
        self.planeWidgetY.SetPicker(self.picker)
        self.planeWidgetY.RestrictPlaneToVolumeOn()
        self.planeWidgetY.TextureInterpolateOff()
        self.planeWidgetY.SetResliceInterpolateToLinear()
        self.planeWidgetY.SetInputData(self.reader.GetOutput())
        self.planeWidgetY.SetPlaneOrientationToYAxes()
        self.planeWidgetY.SetSliceIndex(y)
        self.planeWidgetY.GetTexturePlaneProperty().SetOpacity(1)
        self.planeWidgetY.On()

        self.planeWidgetZ.SetInteractor(self.iren3)
        self.planeWidgetZ.SetPicker(self.picker)
        self.planeWidgetZ.TextureInterpolateOff()
        self.planeWidgetZ.RestrictPlaneToVolumeOn()
        self.planeWidgetZ.SetResliceInterpolateToLinear()
        self.planeWidgetZ.SetInputData(self.reader.GetOutput())
        self.planeWidgetZ.SetPlaneOrientationToZAxes()
        self.planeWidgetZ.SetSliceIndex(z)
        self.planeWidgetZ.GetTexturePlaneProperty().SetOpacity(1)
        self.planeWidgetZ.On()

        self.imgAct1 = vtk.vtkImageActor()
        self.imgAct1.SetInputData(self.planeWidgetX.GetResliceOutput())
        self.ren1.AddActor(self.imgAct1)
        self.imgAct2 = vtk.vtkImageActor()
        self.imgAct2.SetInputData(self.planeWidgetY.GetResliceOutput())
        self.ren2.AddActor(self.imgAct2)
        self.imgAct3 = vtk.vtkImageActor()
        self.imgAct3.SetInputData(self.planeWidgetZ.GetResliceOutput())
        self.ren3.AddActor(self.imgAct3)

    def _open_dcm(self):
        self._path = QtGui.QFileDialog.getExistingDirectory()
        self._path += '\\'
        sub_win = splitWindow(parent=self, path=self._path)
        self.thickness = sub_win.reader.getSliceThcikness()
        self.ui.buttonML.setEnabled(True)
        self.ui.buttonLevelset.setEnabled(True)
        self.ui.buttonMC.setEnabled(True)
        self.ui.buttonCRF.setEnabled(True)
        self.ui.labelName.setText(sub_win.ui.labelName.text())
        self.ui.labelSex.setText(sub_win.ui.labelSex.text())
        self.ui.labelID.setText(sub_win.ui.labelID.text())
        self.ui.logging.append('DCM Files Directory: ' + self._path)
        # 子窗口关闭前阻塞父窗口
        sub_win.setWindowModality(QtCore.Qt.WindowModal)
        sub_win.show()

    def setMarchingCubesValue(self):
        # 打开dicom文件之前拖滑块无效
        if self.iren is None or self._path is None:
            return None
        # 重新渲染
        self.ui.logging.append('Reconstructing...')
        self.mc.SetValue(0, self.ui.mc_spin.value())
        self.iren.Initialize()
        self.ui.logging.append('Reconstruction Finished')

    def get_paths(self):
        return self._path

    def printLog(self, msg):
        self.ui.logging.append(msg)

    def setProgressBar(self, val):
        self.ui.progressBar.setValue(val)

    def updateAxisX(self):
        if self.iren1 is  not None:
            self.planeWidgetX.SetSliceIndex(self.ui.spinBox.value())
            self.iren1.Initialize()

    def updateAxisY(self):
        if self.iren2 is not None:
            self.planeWidgetY.SetSliceIndex(self.ui.spinBox1.value())
            self.iren2.Initialize()

    def updateAxisZ(self):
        if self.iren3 is not None:
            self.planeWidgetZ.SetSliceIndex(self.ui.spinBox2.value())
            self.iren3.Initialize()

    def export2stl(self):
        if self.iren is None:
            return None
        fileName = QtGui.QFileDialog.getSaveFileName(self, 'Save as STL')
        self.ui.logging.append('Save as .stl file \npath: ' + fileName)
        writer = vtk.vtkSTLWriter()
        try:
            writer.SetFileName(fileName)
            writer.SetInputData(self.mc.GetOutput())
            writer.Update()
        except:
            return None

    def export2mhd(self):
        if self.iren is None:
            return None
        fileName = QtGui.QFileDialog.getSaveFileName(self, 'Save as MHD')
        self.mhd_writer = vtk.vtkMetaImageWriter()
        self.mhd_writer.SetFileName(fileName)
        self.mhd_writer.SetRAWFileName(fileName.split('.')[0] + '.raw')
        self.mhd_writer.SetInputData(self.reader.GetOutput())
        self.mhd_writer.Write()
        self.ui.logging.append('Save as *.mhd file \npath: ' + fileName)
        self.ui.logging.append('Save as *.raw file \npath: ' + fileName.split('.')[0] + '.raw')

    def _MarchingCubes(self):
        mcVal, flag = QtGui.QInputDialog.getInt(self, 'INPUT', 'Input MC value here', min=0, max=1000, step=1)
        if self.iren is not None or flag == False:
            return None
        self.ui.mc_slide.setValue(mcVal)
        try:
            fileNames = os.listdir(self._path)
        except:
            QtGui.QMessageBox.warning(self, 'ERROR', 'No Dicom Files to be Shown', QtGui.QMessageBox.Ok)
            return None
        for name in fileNames:
            if '.dcm' not in name:
                self._path = self._path + name + '\\'
                break
        time_1 = datetime.now()
        self._construct_pipeline(self._path, mc=mcVal)
        self.iren.Initialize()
        if self.iren1 is None:
            self._threeViewPipeline(self.ui.spinBox.value(), self.ui.spinBox1.value(), self.ui.spinBox2.value())
            self.iren1.Initialize()
            self.iren2.Initialize()
            self.iren3.Initialize()
        self.ui.logging.append('Time Cost: ' + str(datetime.now() - time_1))

    def _levelset(self):
        self.ui.progressBar.setVisible(True)
        self.ui.progressBar.setRange(0, 100)
        if self._path is None:
            return None
        self.ui.mc_slide.setValue(10)
        fileNames = os.listdir(self._path)
        for name in fileNames:
            if '.dcm' not in name:
                self._path = self._path + '\\' + name + '\\'
                break
        if self.iren is None:
            # Initialize thread
            self.thread_levelset = levelsetThread(path=self._path, parent=self)
            self.thread_levelset.signal.connect(self.setProgressBar)
            self.thread_levelset.send_msg.connect(self.printLog)
            self.thread_levelset.finished.connect(self._levelset_pipeline)
            self.thread_levelset.start()

    def _MachineLearning(self):
        self.ui.progressBar.setVisible(True)
        self.ui.progressBar.setRange(0, 100)
        if self._path is None:
            return None
        self.ui.mc_slide.setValue(10)
        fileNames = os.listdir(self._path)
        for name in fileNames:
            if '.dcm' not in name:
                self._path = self._path + '\\' + name + '\\'
                break
        if self.iren is None:
            # Initialize thread
            self.dt_thread = decisionTreeThread(path=self._path, parent=self)
            self.dt_thread.signal.connect(self.setProgressBar)
            self.dt_thread.send_msg.connect(self.printLog)
            self.dt_thread.finished.connect(self._levelset_pipeline)
            self.dt_thread.start()
        if self.iren1 is None:
            self._threeViewPipeline(self.ui.spinBox.value(), self.ui.spinBox1.value(), self.ui.spinBox2.value())
            self.iren1.Initialize()
            self.iren2.Initialize()
            self.iren3.Initialize()

    def _crf_as_rnn(self):
        self.ui.progressBar.setVisible(True)
        self.ui.progressBar.setRange(0, 100)
        if self._path is None:
            return None
        self.ui.mc_slide.setValue(10)
        fileNames = os.listdir(self._path)
        for name in fileNames:
            if '.dcm' not in name:
                self._path = self._path + '\\' + name + '\\'
                break
        if self.iren is None:
            # Initialize thread
            self.thread_levelset = crfThread(path=self._path, parent=self)
            self.thread_levelset.signal.connect(self.setProgressBar)
            self.thread_levelset.send_msg.connect(self.printLog)
            self.thread_levelset.finished.connect(self._levelset_pipeline)
            self.thread_levelset.start()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())