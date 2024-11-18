import sys
import os
import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from territorytools.process import import_all_data, valid_dir, find_territory_files
from territorytools.urine import Peetector
from territorytools.ttclasses import dict_to_yaml, load_metadata


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
            self.ptect = Peetector(t_files['thermal.avi'], t_files['thermal.h5'])
            self.optical_vid = cv2.VideoCapture(t_files['top.mp4'])
            self.metadata = load_metadata(t_files['metadata.yaml'])
            print(self.get_info())


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
            case 'cool_thresh':
                self.ptect.cool_thresh = value
            case 'walls':
                self.ptect.dead_zones = []
                self.ptect.add_dz(zone=value)


    def get_info(self):
        return dict_to_yaml(self.metadata)


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


    def update_gui(self):
        for c in self.thresh_controls:
            param, val = c.get_value()
            self.control.set_param(param, val)

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


class PtectApp:
    def __init__(self, data_folder: str = None):
        app = QApplication(sys.argv)
        gui = PtectGUI(data_folder=data_folder)
        sys.exit(app.exec())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = PtectGUI()
    sys.exit(app.exec())

