import sys
import os
import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets

from territorytools.process import import_all_data, valid_dir, find_territory_files
from territorytools.urine import Peetector
from territorytools.ttclasses import MDcontroller
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt


matplotlib.use('QT5Agg')

def get_data_dialog():
    dialog = QFileDialog()
    data_fold = dialog.getExistingDirectory()
    return data_fold


class PtectController:
    def __init__(self, data_folder: str = None):
        if data_folder is None:
            data_folder = get_data_dialog()

        self.frame_num = 0
        self.valid = valid_dir(data_folder)
        if self.valid:
            t_files = find_territory_files(data_folder)
            self.optical_vid = cv2.VideoCapture(t_files['top.mp4'])
            self.metadata = MDcontroller(t_files['metadata.yaml'])
            therm_cent = self.metadata.get_val('Territory/thermal_center')
            t_px_per_cm = self.metadata.get_val('Territory/thermal_px_per_cm')
            self.ptect = Peetector(t_files['thermal.avi'], t_files['thermal.h5'], cent_xy=therm_cent, px_per_cm=t_px_per_cm)

    def set_frame(self, frame_num: int):
        if frame_num > self.optical_vid.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_num = 0
        self.frame_num = frame_num

    def read_next_frame(self, *args):
        resize_w, resize_h = 1280, 480
        if len(args) == 2:
            resize_w, resize_h = args[:2]

        c_im = np.empty((resize_h, resize_w, 3))
        self.optical_vid.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
        ret, frame = self.optical_vid.read()
        if ret:
            urine_data, u_frame = self.ptect.peetect_frames(start_frame=self.frame_num,
                                                            num_frames=1, return_frame=True)
            rs_raw = cv2.resize(frame, (resize_w // 2, resize_h))
            rs_urine = cv2.resize(u_frame, (resize_w // 2, resize_h))
            c_im = np.hstack((rs_raw, rs_urine))
        self.set_frame(self.frame_num + 1)
        return c_im.astype('uint8')

    def set_param(self, param, value):
        match param:
            case 'heat_thresh':
                self.ptect.heat_thresh = value
                self.metadata.set_key_val('Territory/ptect_heat_thresh', value)
            case 'cool_thresh':
                self.ptect.cool_thresh = value
                self.metadata.set_key_val('Territory/ptect_cool_thresh', value)
            case 'deadzone':
                self.ptect.add_dz(num_pts=20)
            case 'frame_type':
                self.ptect.frame_type = value

    def get_info(self):
        return str(self.metadata)


class PtectGUI(QWidget):
    playing = False
    thresh_controls = []

    def __init__(self, *args, data_folder: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize(1280, 960)
        self.setWindowTitle('Ptect Preview GUI')
        icon_path = os.path.abspath('../resources/ptect_icon.png')
        self.setWindowIcon(QIcon(icon_path))

        self.preview_frame_w = 1280
        self.preview_frame_h = 480
        self.control = PtectController(data_folder=data_folder)
        first_frame = self.control.read_next_frame(self.preview_frame_w, self.preview_frame_h)

        self.layout = QGridLayout()
        self.prev_frame = QLabel()
        prev_im = QImage(first_frame,
                         self.preview_frame_w,
                         self.preview_frame_h,
                         3 * self.preview_frame_w,
                         QImage.Format_BGR888)
        self.prev_frame.setPixmap(QPixmap(prev_im))
        self.prev_frame.setScaledContents(True)
        self.layout.addWidget(self.prev_frame, 0, 0, 2, 2)

        self.add_controls()

        self.setLayout(self.layout)
        self.prev_frame.show()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(100)

        self.show()

    def add_controls(self):
        params = ['heat_thresh', 'cool_thresh']
        for i, p in zip(range(2, 4), params):
            slider = SlideInputer(p)
            self.layout.addWidget(slider, 0, i, 1, 1)
            self.thresh_controls.append(slider)

        play = QPushButton('Play')
        play.setCheckable(True)

        def play_video():
            if play.isChecked():
                self.playing = True
            else:
                self.playing = False

        play.clicked.connect(play_video)
        self.layout.addWidget(play, 2, 0, 1, 1)
        info_gb = QGroupBox('Run Info')
        info_box = QVBoxLayout()
        self.run_info = QLabel(self.control.get_info())
        info_box.addWidget(self.run_info)
        info_gb.setLayout(info_box)
        self.layout.addWidget(info_gb, 0, 5, 1, 1)

        dz_but = QPushButton('Add Dead Zone')
        dz_but.clicked.connect(self.set_dz)
        self.layout.addWidget(dz_but, 1, 3, 1, 1)
        set_frame = QCheckBox('Show Steps')
        def set_frame_type():
            if set_frame.isChecked():
                self.control.set_param('frame_type', 2)
            else:
                self.control.set_param('frame_type', 0)
        set_frame.clicked.connect(set_frame_type)
        self.layout.addWidget(set_frame, 1, 2, 1, 1)
        mplw = MplWidget()
        self.layout.addWidget(mplw, 3, 0, 1, 1)

    def set_dz(self):
        self.control.set_param('deadzone', 'na')

    def update_gui(self):
        for c in self.thresh_controls:
            param, val = c.get_value()
            self.control.set_param(param, val)

        self.run_info.setText(self.control.get_info())

        if self.playing:
            frame = self.control.read_next_frame(self.preview_frame_w, self.preview_frame_h)
            w = frame.shape[1]
            h = frame.shape[0]
            q_frame = QImage(frame, w, h, 3 * w, QImage.Format_BGR888)
            self.prev_frame.setPixmap(QPixmap(q_frame))


class SlideInputer(QGroupBox):
    def __init__(self, name):
        super().__init__(name)
        self.id = name
        slide_group = QVBoxLayout()
        self.slide = QSlider()
        self.slide.setMinimum(0)
        self.slide.setMaximum(255)
        self.slide.valueChanged.connect(self.update_ebox)

        self.ebox = QLineEdit()
        self.ebox.setValidator(QIntValidator())
        self.ebox.setMaxLength(3)
        self.ebox.textChanged.connect(self.update_slide)

        for w in (self.slide, self.ebox):
            slide_group.addWidget(w)
        self.setLayout(slide_group)

    def update_slide(self, val):
        if len(val) > 0:
            val = int(val)
            self.slide.setValue(val)

    def update_ebox(self, val):
        val = str(val)
        self.ebox.setText(val)

    def get_value(self):
        return self.id, self.slide.value()


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111)
        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

# Matplotlib widget
class MplWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        self.vbl = QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)


class PtectApp:
    def __init__(self, data_folder: str = None):
        app = QApplication(sys.argv)
        gui = PtectGUI(data_folder=data_folder)
        sys.exit(app.exec())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = PtectGUI()
    sys.exit(app.exec())
