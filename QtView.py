import sys, os, inspect
import vtk
from PyQt4 import QtCore, QtGui
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

'''
    这个文件没有什么卵用，只是记录下原来用vtkImageData时候的代码是怎么写的
    方便以后移植levelset和机器学习的结果
'''

dicom_path = 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\MLLS\\data\\dao_series1\\'
new_data_path1 = 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\new_data\\series1\\'
new_data_path2 = 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\new_data\\series2\\'
new_path = 'E:\\C++\\Projects\\surgeryGuidingProject_copy\\8848\\new\DICOM\\20170510\\08290000\\'

class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.frame = QtGui.QFrame()
        self.vl = QtGui.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(dicom_path)
        self.reader.Update()

        dims = self.reader.GetOutput().GetDimensions()
        origin = self.reader.GetOutput().GetOrigin()
        self.readerImageCast = vtk.vtkImageCast()
        self.readerImageCast.SetInputConnection(self.reader.GetOutputPort())
        self.readerImageCast.SetOutputScalarTypeToFloat()
        self.readerImageCast.Update()

        print(dims); print(origin)
        self.data = vtk.vtkImageData()
        self.data.SetDimensions(dims[0], dims[1], dims[2])
        self.data.AllocateScalars(vtk.VTK_FLOAT, 1)
        self.data.SetSpacing(1, 1, 0.625)
        self.data.SetOrigin(origin[0], origin[1], origin[2])
        print(self.data.GetActualMemorySize())

        self.mc = vtk.vtkMarchingCubes()
        # self.mc.SetInputData(self.data)
        self.mc.SetInputConnection(self.readerImageCast.GetOutputPort())
        self.mc.SetValue(0, 320)
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

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

    def export2stl(self, fileName):
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(fileName)
        writer.SetInputData(self.mc.GetOutput())
        writer.Update()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.iren.Initialize()
    # window.export2stl('view.stl')
    sys.exit(app.exec_())