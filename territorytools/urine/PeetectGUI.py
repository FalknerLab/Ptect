# from PyQt5.QtWidgets import QApplication, QLabel, QGridLayout, QLineEdit, QCheckBox, QWidget, QPushButton
# from PyQt5.QtGui import QPixmap, QImage

# class PeetectGUI(QWidget):
#     def __init__(self):
#         super(QWidget, self).__init__()
#         grid = QGridLayout()
#         self.add_top_widgets(grid)
#         self.add_images(grid)
#         self.setLayout(grid)
#         self.show()
#
#     def add_top_widgets(self, grid):
#         c_box = QCheckBox('Walls')
#         grid.addWidget(c_box, 2, 1, 1, 1)
#         grid.addWidget(QLabel('Heat Thresh'),1,2,1,1)
#         grid.addWidget(QLineEdit(),2,2,1,1)
#         grid.addWidget(QLabel('Smoothing Kernel Width'), 1, 3, 1, 1)
#         grid.addWidget(QLineEdit(), 2, 3, 1, 1)
#         grid.addWidget(QLabel('Dilation Kernel Width'), 1, 4, 1, 1)
#         grid.addWidget(QLineEdit(), 2, 4, 1, 1)
#         grid.addWidget(QPushButton('Play'), 2, 5, 1, 1)
#
#     def add_images(self, grid):
#         im = QImage(np.zeros((640,480)),640,480, 3 * 640, QImage.Format_RGB888)
#         lab = QLabel()
#         lab.setPixmap(QPixmap.fromImage(im))
#         grid.addWidget(lab, 3, 1, 1, 2)
#         im2 = QImage(np.zeros((640, 480)), 640, 480, 3 * 640, QImage.Format_RGB888)
#         lab = QLabel()
#         lab.setPixmap(QPixmap.fromImage(im))
#         grid.addWidget(lab, 3, 3, 1, 2)
#         im3 = QImage(np.zeros((640, 480)), 640, 480, 3 * 640, QImage.Format_RGB888)
#         lab = QLabel()
#         lab.setPixmap(QPixmap.fromImage(im))
#         grid.addWidget(lab, 3, 5, 1, 2)