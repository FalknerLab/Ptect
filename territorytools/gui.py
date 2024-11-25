import sys
import os
import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from numpy.ma.core import minimum

from territorytools.process import import_all_data, valid_dir, find_territory_files
from territorytools.urine import Peetector
from territorytools.ttclasses import MDcontroller
from territorytools.plotting import add_territory_circle
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt


matplotlib.use('QT5Agg')
plt.ion()

def get_data_dialog():
    dialog = QFileDialog()
    data_fold = dialog.getExistingDirectory()
    return data_fold


class PtectController:
    data_buf = []
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
            self.therm_vid = t_files['thermal.avi']
            self.ptect = Peetector(self.therm_vid, t_files['thermal.h5'], cent_xy=therm_cent, px_per_cm=t_px_per_cm)
            self.test_frame = self.ptect.read_frame(0)[1]

    def set_frame(self, frame_num: int):
        if frame_num > self.optical_vid.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_num = 0
        self.frame_num = frame_num

    def get_data(self):
        times = None
        hot_marks = None
        if len(self.data_buf[0]) > 0:
            times = self.data_buf[0][:, 0]
            hot_marks = self.data_buf[0][:, 1:]
        return times, hot_marks

    def read_next_frame(self, *args):
        resize_w, resize_h = 1280, 480
        if len(args) == 2:
            resize_w, resize_h = args[:2]

        c_im = np.empty((resize_h, resize_w, 3))
        self.optical_vid.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
        ret, frame = self.optical_vid.read()
        out_frame = None
        if ret:
            urine_data, out_frame = self.ptect.peetect_frames(start_frame=self.frame_num,
                                                            num_frames=1, return_frame=True)
            rs_raw = cv2.resize(frame, (resize_w // 2, resize_h))
            rs_urine = cv2.resize(out_frame, (resize_w // 2, resize_h))
            c_im = np.hstack((rs_raw, rs_urine))
            self.data_buf = urine_data
        self.set_frame(self.frame_num + 1)
        return c_im.astype('uint8'), frame, out_frame

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
            case 'arena':
                if value[0] == 'circle' or value[0] == 'rectangle':
                    self.ptect.arena_cnt[0] = value[1][0]
                    self.ptect.arena_cnt[1] = value[1][1]
                    self.ptect.set_valid_arena(value[0], *value[1][2:])
                else:
                    self.ptect.set_valid_arena(value[0], *value[1])

    def get_info(self):
        return str(self.metadata)


class PtectGUI(QWidget):
    playing = False
    thresh_controls = []

    def __init__(self, *args, data_folder: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize(1280, 1400)
        self.setWindowTitle('Ptect Preview GUI')
        icon_path = os.path.abspath('../resources/ptect_icon.png')
        self.setWindowIcon(QIcon(icon_path))

        self.preview_frame_w = 1280
        self.preview_frame_h = 480
        self.control = PtectController(data_folder=data_folder)
        first_frames = self.control.read_next_frame(self.preview_frame_w, self.preview_frame_h)
        self.prev_im = first_frames[2]

        self.layout = QGridLayout()
        self.prev_frame = QLabel()
        q_im = QImage(first_frames[0],
                         self.preview_frame_w,
                         self.preview_frame_h,
                         3 * self.preview_frame_w,
                         QImage.Format_BGR888)
        self.prev_frame.setPixmap(QPixmap(q_im))
        self.prev_frame.setScaledContents(True)
        self.layout.addWidget(self.prev_frame, 0, 0, 4, 3)

        self.add_controls()

        self.setLayout(self.layout)
        self.prev_frame.show()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(100)

        self.show()

    def add_controls(self):
        params = ['heat_thresh', 'cool_thresh']
        num_slides = len(params)
        end_slides = 3+num_slides
        for i, p in zip(range(3, end_slides), params):
            slider = SlideInputer(p)
            self.layout.addWidget(slider, 3, i, 1, 1)
            self.thresh_controls.append(slider)

        play = QPushButton('Play')
        play.setCheckable(True)
        def play_video():
            if play.isChecked():
                self.playing = True
            else:
                self.playing = False
        play.clicked.connect(play_video)
        self.layout.addWidget(play, 5, 0, 1, 1)

        load_but = QPushButton('Load Folder')
        def load_data():
            print('yo im loadin the data')
        load_but.clicked.connect(load_data)
        self.layout.addWidget(load_but, 5, 1, 1, 1)

        info_gb = QGroupBox('Run Info')
        info_box = QVBoxLayout()
        self.run_info = QLabel(self.control.get_info())
        info_box.addWidget(self.run_info)
        info_gb.setLayout(info_box)
        self.layout.addWidget(info_gb, 3, end_slides+1, 1, 1)

        dz_but = QPushButton('Add Dead Zone')
        dz_but.clicked.connect(self.set_dz)
        self.layout.addWidget(dz_but, 2, 4, 1, 1)

        set_frame = QCheckBox('Show Steps')
        def set_frame_type():
            if set_frame.isChecked():
                self.control.set_param('frame_type', 2)
            else:
                self.control.set_param('frame_type', 0)
        set_frame.clicked.connect(set_frame_type)
        self.layout.addWidget(set_frame, 2, 3, 1, 1)

        self.plotter = PlotWidget()
        add_territory_circle(self.plotter.gca())
        self.layout.addWidget(self.plotter, 4, 0, 1, 1)

        plotter = PlotWidget()
        add_territory_circle(plotter.gca())
        self.layout.addWidget(plotter, 4, 1, 1, 1)

        self.arena_controls = ArenaSelector('Arena Controls')
        self.arena_controls.set_frame(self.control.test_frame)
        self.layout.addWidget(self.arena_controls, 0, 3, 2, num_slides+3)

    def set_dz(self):
        self.control.set_param('deadzone', 'na')

    def update_gui(self):
        for c in self.thresh_controls:
            param, val = c.get_value()
            self.control.set_param(param, val)

        self.run_info.setText(self.control.get_info())

        arena_params = self.arena_controls.get_arena_data()
        self.control.set_param('arena', arena_params)

        if self.playing:
            # times, hot_marks = self.control.get_data()
            # if times is not None:
            #     self.plotter.plot(hot_marks[:, 0], hot_marks[:, 1], plot_style='scatter', c=times)
            frames = self.control.read_next_frame(self.preview_frame_w, self.preview_frame_h)
            frame = frames[0]
            self.prev_im = frames[2]
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

class ArenaSelector(QGroupBox):
    size = 0
    sub_controls = []
    frame_buf = None
    custom_pts = []
    def __init__(self, name, arena_type='circle'):
        super().__init__(name)
        self.arena_type = arena_type
        self.layout = QGridLayout()
        list_wid = QComboBox()
        list_wid.addItems(['circle', 'rectangle', 'custom'])
        list_wid.currentTextChanged.connect(self.update_controls)
        self.update_controls(self.arena_type)
        self.layout.addWidget(list_wid, 0, 0, 1, 3)
        self.setLayout(self.layout)


    def update_controls(self, arena):
        for sc in self.sub_controls:
            sc.setParent(None)
        self.sub_controls = []
        self.arena_type = arena
        match arena:
            case 'circle':
                x_slide = QSlider(Qt.Orientation.Horizontal, minimum=1, maximum=1000)
                y_slide = QSlider(Qt.Orientation.Horizontal, minimum=1, maximum=1000)
                slide = QSlider(Qt.Orientation.Horizontal, minimum=1, maximum=1000)
                label = QLabel('Radius')
                self.sub_controls.append(x_slide)
                self.sub_controls.append(y_slide)
                self.layout.addWidget(x_slide, 1, 0, 1, 1)
                self.layout.addWidget(y_slide, 2, 0, 1, 1)
                self.sub_controls.append(slide)
                self.sub_controls.append(label)
                self.layout.addWidget(slide, 1, 2, 1, 1)
                self.layout.addWidget(label, 1, 1, 1, 1)
            case 'rectangle':
                x_slide = QSlider(Qt.Orientation.Horizontal, minimum=1, maximum=1000)
                y_slide = QSlider(Qt.Orientation.Horizontal, minimum=1, maximum=1000)
                self.sub_controls.append(x_slide)
                self.sub_controls.append(y_slide)
                self.layout.addWidget(x_slide, 1, 0, 1, 1)
                self.layout.addWidget(y_slide, 2, 0, 1, 1)
                labs = ['Width', 'Height']
                for i in range(2):
                    slide = QSlider(Qt.Orientation.Horizontal, minimum=1, maximum=1000)
                    self.sub_controls.append(slide)
                    self.layout.addWidget(slide, i+1, 2, 1, 1)
                    label = QLabel(labs[i])
                    self.sub_controls.append(label)
                    self.layout.addWidget(label, i+1, 1, 1, 1)
            case 'custom':
                draw_but = QPushButton('Draw Arena')
                draw_but.clicked.connect(self.draw_arena)
                self.sub_controls.append(draw_but)
                self.layout.addWidget(draw_but)

    def get_arena_data(self):
        outs_args = []
        if self.arena_type != 'custom':
            for sc in self.sub_controls:
                if isinstance(sc, QSlider):
                    outs_args.append(sc.value())
        else:
            outs_args = self.custom_pts
        return self.arena_type, outs_args


    def draw_arena(self):
        plt.figure()
        plt.imshow(self.frame_buf)
        pnts = plt.ginput(n=6, timeout=600)
        self.custom_pts = pnts

    def set_frame(self, frame):
        self.frame_buf = frame

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111)
        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        pyqt_grey = (240/255, 240/255, 240/255)
        self.canvas.fig.set_facecolor(pyqt_grey)
        self.canvas.ax.set_facecolor(pyqt_grey)
        self.canvas.ax.spines['top'].set_visible(False)
        self.canvas.ax.spines['right'].set_visible(False)
        self.vbl = QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)

    def gca(self):
        return self.canvas.ax

    def plot(self, *args, plot_style=None, **kwargs):
        my_ax = self.gca()
        if plot_style is not None:
            if plot_style == 'scatter':
                my_ax.scatter(args[0], args[1], **kwargs)
        else:
            my_ax.plot(*args, **kwargs)
        self.canvas.draw()


class PtectApp:
    def __init__(self, data_folder: str = None):
        app = QApplication(sys.argv)
        gui = PtectGUI(data_folder=data_folder)
        sys.exit(app.exec())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = PtectGUI()
    sys.exit(app.exec())
