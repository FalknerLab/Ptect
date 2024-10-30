import sys
import os
import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from territorytools.process import import_all_data, valid_dir, find_territory_files
from territorytools.urine import Peetector


def get_data_dialog():
    dialog = QFileDialog()
    data_fold = dialog.getExistingDirectory()
    return data_fold


class PtectGUI(QMainWindow):
    def __init__(self, parent=None, data_folder: str = None):
        app = QApplication([])
        super(PtectGUI, self).__init__(parent)
        if data_folder is None:
            data_folder = get_data_dialog()
        if valid_dir(data_folder):
            self.resize(1280, 960)
            self.setWindowTitle('Ptect Preview GUI')
            icon_path = os.path.abspath('../resources/ptect_icon.png')
            self.setWindowIcon(QIcon(icon_path))
            t_files = find_territory_files(data_folder)
            self.vid_cap_list = []
            self.vid_cap_list.append(cv2.VideoCapture(t_files[2]))
            first_frame = self.make_prev_frame()
            w = first_frame.shape[1]
            h = first_frame.shape[0]

            self.ptect = Peetector(t_files[2], t_files[3])

            layout = QGridLayout()
            self.prev_frame = QLabel()
            prev_im = QImage(first_frame, w, h, 3 * w, QImage.Format_BGR888)
            self.prev_frame.setPixmap(QPixmap(prev_im))
            layout.addWidget(self.prev_frame, 0, 0)
            cent_wid = QWidget()
            cent_wid.setLayout(layout)
            self.setCentralWidget(cent_wid)
            self.prev_frame.show()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_gui)
            self.timer.start(25)
            self.show()
        self.frame_num = 0
        sys.exit(app.exec_())

    def update_gui(self):
        frame = self.make_prev_frame()
        w = frame.shape[1]
        h = frame.shape[0]
        q_frame = QImage(frame, w, h, 3 * w, QImage.Format_BGR888)
        self.prev_frame.setPixmap(QPixmap(q_frame))

    def make_prev_frame(self):
        c_im = np.empty((480, 0, 3))
        for v in self.vid_cap_list:
            v.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
            ret, frame = v.read()
            if ret:
                urine_data = self.ptect.peetect_frames()
                rs_frame = cv2.resize(frame, (640, 480))
            else:
                rs_frame = np.zeros((480, 640, 3))
            c_im = np.hstack((c_im, rs_frame))
        return c_im.astype('uint8')


if __name__ == '__main__':
    gui = PtectGUI()
