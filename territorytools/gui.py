import sys
import os
import cv2
import h5py
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from matplotlib.collections import PathCollection
from territorytools.process import process_all_data, valid_dir, find_territory_files
from territorytools.urine import Peetector, PtectPipe
from territorytools.ttclasses import MDcontroller
from territorytools.plotting import add_territory_circle
from territorytools.behavior import get_territory_data
from territorytools.utils import intersect2d
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt


matplotlib.use('QT5Agg')

# Plotting Globals
MOUSE_COLORS_BGR = []
MOUSE_COLORS_MPL = ('tab:blue', 'tab:orange')
for c in MOUSE_COLORS_MPL:
    MOUSE_COLORS_BGR.append(255*np.fliplr(np.array(matplotlib.colors.to_rgb(c))[None, :])[0])

def get_data_dialog():
    dialog = QFileDialog()
    data_fold = dialog.getExistingDirectory()
    return data_fold

def get_save_dialog():
    dialog = QFileDialog()
    save_path = dialog.getSaveFileName()
    return save_path[0]


class PtectController:
    data_buf = []
    hot_data = np.empty((0, 3))
    cool_data = np.empty((0, 2))
    def __init__(self, data_folder: str = None):
        if data_folder is None:
            data_folder = get_data_dialog()
        self.data_folder = data_folder
        self.test = 0
        self.frame_num = 0
        self.t_frame = 0
        self.control_hz = 0
        self.save_path = None
        self.valid = valid_dir(data_folder)
        if self.valid:
            t_files = find_territory_files(data_folder)
            self.metadata = MDcontroller(t_files['ptmetadata.yml'])
            self.op_hz = self.metadata.get_val('Territory/optical_hz')
            self.t_offset_frames = self.metadata.get_val('Territory/thermal_offset')
            self.t_frame = self.t_offset_frames
            self.therm_hz = self.metadata.get_val('Territory/thermal_hz')
            orient = self.metadata.get_val('Territory/orientation')
            self.control_hz = max(self.op_hz, self.therm_hz)
            self.optical_vid = cv2.VideoCapture(t_files['top.mp4'])
            sleap_file = h5py.File(t_files['fixedtop.h5'], 'r')
            self.sleap_data = sleap_file['tracks']
            self.optical_data = []
            for i in self.sleap_data:
                this_data = get_territory_data(i, rot_offset=orient,
                                               px_per_cm=self.metadata.get_val('Territory/optical_px_per_cm'),
                                               ref_point=self.metadata.get_val('Territory/optical_center'),
                                               hz = self.metadata.get_val('Territory/optical_hz'))
                self.optical_data.append(this_data)
            therm_cent = self.metadata.get_val('Territory/thermal_center')
            t_px_per_cm = self.metadata.get_val('Territory/thermal_px_per_cm')
            self.therm_vid = t_files['thermal.avi']
            self.ptect = Peetector(self.therm_vid, t_files['thermal.h5'], cent_xy=therm_cent, px_per_cm=t_px_per_cm,
                                   rot_ang=orient, start_frame=self.t_frame)
            self.test_frame = self.ptect.read_frame()[1]
            dz_data = self.metadata.get_val('Territory/deadzone')
            if len(dz_data) > 0:
                self.set_param('deadzone', dz_data)
            self.last_t_frame = self.test_frame

    def set_frame(self, frame_num: int):
        if frame_num > self.optical_vid.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_num = 0
        self.frame_num = frame_num
        self.ptect.set_frame(frame_num)

    def get_data(self, which_data, time_win=None):
        start_ind = self.frame_num - 1
        if time_win is not None:
            start_ind = self.frame_num - time_win
            start_ind = max(start_ind, 0)

        if which_data == 'therm_len':
            return self.ptect.get_length()

        if which_data == 'length':
            return self.optical_vid.get(cv2.CAP_PROP_FRAME_COUNT)

        if which_data == 'thermal':
            data_cop = np.copy(self.hot_data)
            # data_cop[:, 0] = np.round(self.control_hz*(data_cop[:, 0] / self.therm_hz))
            is_self = data_cop[:, 1] < 0
            self_marks = data_cop[is_self, :]
            other_marks = data_cop[~is_self, :]
            return self.hot_data, (self_marks, other_marks)

        if which_data == 'velocity':
            xys = []
            for d in self.optical_data:
                xys.append(d[3])
            return xys

        if which_data == 'optical':
            xys = []
            for d in self.optical_data:
                xys.append((d[0][start_ind:self.frame_num], d[1][start_ind:self.frame_num]))
            return xys
        else:
            return None

    def get_optical_frame(self):
        op_frame_num = round(self.frame_num * (self.op_hz / self.control_hz))
        if op_frame_num != self.frame_num:
            self.optical_vid.set(cv2.CAP_PROP_POS_FRAMES, op_frame_num)
        ret, frame = self.optical_vid.read()
        for ind, sd in enumerate(self.sleap_data):
            not_nan = ~np.isnan(sd[0, :, op_frame_num])
            good_pts = sd[:, not_nan, op_frame_num].astype(int)
            for x, y in zip(good_pts[0, :], good_pts[1, :]):
                cv2.circle(frame, (x, y), 5, MOUSE_COLORS_BGR[ind], -1)
        return ret, frame

    def read_next_frame(self, *args):
        resize_w, resize_h = 1280, 480
        if len(args) == 2:
            resize_w, resize_h = args[:2]

        c_im = np.empty((resize_h, resize_w, 3))
        ret, frame = self.get_optical_frame()
        rs_raw = cv2.resize(frame, (resize_w // 2, resize_h))
        out_frame = None
        last_frame = self.t_frame
        new_frame = round(self.frame_num * (self.therm_hz / self.control_hz)) + self.t_offset_frames
        if new_frame > last_frame:
            urine_data, out_frame, f_num = self.ptect.peetect_next_frame(return_frame=True)
            rs_urine = cv2.resize(out_frame, (resize_w // 2, resize_h))
            self.update_data(urine_data)
            self.t_frame = new_frame
            self.last_t_frame = out_frame
        else:
            rs_urine = cv2.resize(self.last_t_frame, (resize_w // 2, resize_h))
        c_im = np.hstack((rs_raw, rs_urine))
        self.set_frame(self.frame_num + 1)
        return c_im.astype('uint8'), frame, out_frame

    def update_data(self, data):
        if len(data[0]) > 0:
            this_d = data[0]
            this_d[:, 0] = self.frame_num
            new_marks, new_inds = intersect2d(self.hot_data[:, 1:], this_d[:, 1:], return_int=False)
            self.hot_data = np.vstack((self.hot_data, this_d[new_inds, :]))

    def set_param(self, param, value):
        match param:
            case 'heat_thresh':
                self.ptect.heat_thresh = value
                self.metadata.set_key_val('Territory/ptect_heat_thresh', value)
            case 'cool_thresh':
                self.ptect.cool_thresh = value
                self.metadata.set_key_val('Territory/ptect_cool_thresh', value)
            case 'time_thresh':
                self.ptect.set_time_win(value)
                self.metadata.set_key_val('Territory/ptect_time_thresh', value)
            case 'deadzone':
                if type(value) == list:
                    self.ptect.add_dz(zone=value)
                else:
                    pts = self.ptect.add_dz(num_pts=int(value))
                    self.metadata.set_key_val('Territory/deadzone', pts)
            case 'frame_type':
                self.ptect.frame_type = value
            case 'arena':
                if value[0] == 'circle' or value[0] == 'rectangle':
                    c_x = value[1][0]
                    c_y = value[1][1]
                    self.ptect.arena_cnt[0] = c_x
                    self.ptect.arena_cnt[1] = c_y
                    self.ptect.set_valid_arena(value[0], *value[1][2:])
                    self.metadata.set_key_val('Territory/thermal_center', [c_x, c_y])
                    if value[0] == 'circle':
                        self.metadata.set_key_val('Territory/arena_data', value[1][2])
                    else:
                        self.metadata.set_key_val('Territory/arena_data', value[1][2:])
                else:
                    self.ptect.set_valid_arena(value[0], *value[1])
                    self.metadata.set_key_val('Territory/arena_data', value[1])

    def get_metadata(self, md):
        match md:
            case 'arena':
                return self.metadata.get_val('Territory/thermal_center'), self.metadata.get_val('Territory/arena_type'), self.metadata.get_val('Territory/arena_data')

    def get_info(self):
        return str(self.metadata)

    def save_info(self):
        self.metadata.save_metadata(self.metadata.file_name)

    def run_and_save(self, ppipe):
        self.ptect.run_ptect(pipe=ppipe, save_path=self.save_path, end_frame=1000)
        process_all_data(self.data_folder, skip_ptect=False)

class PtectGUIpipe(PtectPipe):
    def __init__(self, pipe: pyqtSignal(tuple)):
        self.pipe = pipe

    def send(self, *args):
        self.pipe.emit(args[0])


class RunSignals(QObject):
    progress = pyqtSignal(tuple)
    finished = pyqtSignal()


class PtectRunner(QRunnable):
    def __init__(self, ptect_cont: PtectController):
        super().__init__()
        self.signals = RunSignals()
        self.ptect_cont = ptect_cont

    def run(self):
        self.ptect_cont.run_and_save(PtectGUIpipe(self.signals.progress))
        self.signals.finished.emit()

    def set_stop_cb(self, stop_cb):
        self.signals.finished.connect(stop_cb)


class PtectThread(QThreadPool):
    result = pyqtSignal(tuple)

    def __init__(self, ptect_cont: PtectController):
        super().__init__()
        self.ptect_cont = ptect_cont

    def spawn_workers(self, stop_cb=None):
        worker = PtectRunner(self.ptect_cont)
        worker.signals.progress.connect(self.emit_worker_output)
        if stop_cb is not None:
            worker.set_stop_cb(stop_cb)
        self.start(worker)

    def emit_worker_output(self, output):
        self.result.emit(output)


class PtectMainWindow(QMainWindow):
    def __init__(self, data_folder=None):
        super().__init__()
        print('Importing Data...')
        self.control = PtectController(data_folder=data_folder)
        self.preview = PtectPreviewWindow(ptect_cont=self.control, parent=self)
        self.run_win = PtectRunWindow(self.control, parent=self)
        self.preview.show()

    def start_ptect(self):
        self.preview.hide()
        self.run_win.show()
        save_path = get_save_dialog()
        self.control.save_path = save_path
        self.run_win.thread_pool.spawn_workers(stop_cb=self.stop_ptect)

    def stop_ptect(self):
        self.run_win.hide()
        self.preview.show()


class PtectRunWindow(QWidget):
    def __init__(self, ptect_cont, parent=None):
        super().__init__()
        self.parent = parent
        self.control = ptect_cont
        icon_path = os.path.abspath('../resources/ptect_icon.png')
        self.icon = QIcon(icon_path)
        self.setWindowIcon(self.icon)
        res = self.screen().size()
        self.setGeometry(int(res.width()/2 - 60), int(res.height()/2 - 210), 420, 120)
        self.setFixedSize(420, 120)
        self.setWindowIcon(self.icon)
        self.setWindowTitle('Running Ptect...')

        finish_i_path = os.path.abspath('../resources/finish_icon.png')
        self.finish_w = QLabel(self)
        self.finish_w.setPixmap(QPixmap(finish_i_path))
        self.finish_w.move(340, 60)

        self.message = QLabel('Ptecting... On Frame:', self)
        self.message.move(10, 10)
        self.mouse_i_path = os.path.abspath('../resources/mouse_icon.png')
        self.icon_w = QLabel(self)
        self.icon_w.setPixmap(QPixmap(self.mouse_i_path))
        self.icon_w.move(20, 60)

        self.start = 0
        self.stop = 0
        self.thread_pool = PtectThread(self.control)
        self.register_actions()

    def register_actions(self):
        self.thread_pool.result.connect(self.update_run)

    def update_run(self, s):
        end_x = self.finish_w.pos().x()
        frac_done = s[0] / s[1]
        next_x = int(frac_done*end_x)
        orig_p = self.icon_w.pos()
        self.icon_w.move(next_x, orig_p.y())
        self.icon_w.update()

        self.message.clear()
        self.message = QLabel(f'Ptecting... On Frame: {s[0]} of {s[1]}', self)
        self.message.move(10, 10)
        self.message.show()


class PtectPreviewWindow(QWidget):
    playing = False
    thresh_controls = []

    def __init__(self, *args, ptect_cont=None, parent=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.app = parent
        res = self.screen().size()
        self.resize(int(res.width()*0.8), int(res.height()*0.8))
        self.setWindowTitle('Ptect Preview GUI')
        icon_path = os.path.abspath('../resources/ptect_icon.png')
        self.icon = QIcon(icon_path)
        self.setWindowIcon(self.icon)

        self.preview_frame_w = 1960
        self.preview_frame_h = 840

        self.control = ptect_cont

        print('Initializing GUI...')
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
        self.layout.addWidget(self.prev_frame, 0, 0, 3, 2)

        self.add_controls()
        self.set_controls()

        self.setLayout(self.layout)
        self.prev_frame.show()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(10)

        self.draw_timer = QTimer(self)
        self.draw_timer.timeout.connect(self.draw_plots)
        self.draw_timer.start(1000)

    def add_controls(self):
        params = ['heat_thresh', 'cool_thresh', 'time_thresh']
        disp_names = ['Heat Thesh', 'Cool Thresh', 'Check Ahead # of Frames']
        num_slides = len(params)
        end_slides = 2+num_slides
        for i, p, n in zip(range(2, end_slides), params, disp_names):
            slider = SlideInputer(p, label=n)
            self.layout.addWidget(slider, 2, i, 1, 1)
            self.thresh_controls.append(slider)

        play = QPushButton('Play')
        play.setCheckable(True)
        def play_video():
            if play.isChecked():
                self.playing = True
            else:
                self.playing = False
        play.clicked.connect(play_video)

        load_but = QPushButton('Load Folder')
        def load_data():
            self.control = PtectController()
            self.set_controls()
        load_but.clicked.connect(load_data)

        run_but = QPushButton('Run and Save')
        def run_ptect():
            self.app.start_ptect()

        run_but.clicked.connect(run_ptect)
        but_group = QWidget()
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(play)
        sub_layout.addWidget(load_but)
        sub_layout.addWidget(run_but)
        but_group.setLayout(sub_layout)
        self.layout.addWidget(but_group, 5, 0, 1, 1)


        info_gb = QGroupBox('Run Info')
        info_box = QVBoxLayout()
        scroller = QScrollArea()
        self.run_info = QLabel(self.control.get_info())
        scroller.setWidget(self.run_info)
        info_box.addWidget(scroller)
        info_gb.setLayout(info_box)
        self.layout.addWidget(info_gb, 2, end_slides+1, 1, 1)

        dz_but = QPushButton('Add Dead Zone')
        dz_but.clicked.connect(self.set_dz)
        self.layout.addWidget(dz_but, 1, 3, 1, 1)
        self.dz_pt_box = QLineEdit()
        self.dz_pt_box.setValidator(QIntValidator(3, 15))
        self.layout.addWidget(self.dz_pt_box, 1, 4, 1, 1)
        pt_lab = QLabel('# of Points')
        self.layout.addWidget(pt_lab, 1, 5, 1, 1)

        save_info_but = QPushButton('Save Info')
        save_info_but.clicked.connect(self.control.save_info)
        self.layout.addWidget(save_info_but, 1, 6, 1, 1)

        set_frame = QCheckBox('Show Steps')
        def set_frame_type():
            if set_frame.isChecked():
                self.control.set_param('frame_type', 2)
            else:
                self.control.set_param('frame_type', 0)
        set_frame.clicked.connect(set_frame_type)
        self.layout.addWidget(set_frame, 1, 2, 1, 1)

        self.mark_plotter = PlotWidget()
        mark_ax = self.mark_plotter.gca()
        mark_ax.set_title('Detected Marks')
        add_territory_circle(mark_ax, block='block0')
        self.mark_plotter.colorbar(0, self.control.get_data('therm_len'), label='Mark Time (frame)')
        self.layout.addWidget(self.mark_plotter, 3, 1, 2, 1)

        self.xy_plotter = PlotWidget()
        xy_ax = self.xy_plotter.gca()
        xy_ax.set_title('XY Position (cm)')
        add_territory_circle(xy_ax, block='block0')
        xy_ax.plot([0,0], [0,0], c=MOUSE_COLORS_MPL[0], label='Self')
        xy_ax.plot([0,0], [0,0], c=MOUSE_COLORS_MPL[1], label='Other')
        xy_ax.legend()
        self.layout.addWidget(self.xy_plotter, 3, 0, 2, 1)

        self.vel_plotter = PlotWidget()
        vel_ax = self.vel_plotter.gca()
        vel_ax.set_xlim(0, self.control.get_data('length'))
        vel_ax.set_title('Mouse Velocity (normalized)')
        vels = self.control.get_data('velocity')
        vel_ax.plot(vels[0]/max(vels[0]), c=MOUSE_COLORS_MPL[0], label='Self')
        vel_ax.plot(1 + vels[1]/max(vels[1]), c=MOUSE_COLORS_MPL[1], label='Other')
        vel_ax.legend()
        self.layout.addWidget(self.vel_plotter, 3, 2, 1, 6)

        self.raster_plotter = PlotWidget()
        rast_ax = self.raster_plotter.gca()
        rast_ax.set_xlim(0, self.control.get_data('length'))
        rast_ax.set_title('Marking Raster')
        self.layout.addWidget(self.raster_plotter, 4, 2, 1, 6)

        self.arena_controls = ArenaSelector('Arena Controls')
        self.layout.addWidget(self.arena_controls, 0, 2, 1, num_slides+3)

    def set_controls(self):
        arena_data = self.control.get_metadata('arena')
        self.arena_controls.set_values(arena_data)
        self.arena_controls.set_frame(self.control.test_frame)

    def set_dz(self):
        self.control.set_param('deadzone', self.dz_pt_box.text())

    def update_gui(self):
        for c in self.thresh_controls:
            param, val = c.get_value()
            self.control.set_param(param, val)

        self.run_info.setText(self.control.get_info())

        arena_params = self.arena_controls.get_arena_data()
        self.control.set_param('arena', arena_params)
        if self.playing:
            self.update_plots()
            self.update_video()

    def draw_plots(self):
        for p in (self.xy_plotter, self.raster_plotter, self.vel_plotter, self.mark_plotter):
            p.draw()

    def update_video(self):
        frames = self.control.read_next_frame(self.preview_frame_w, self.preview_frame_h)
        frame = frames[0]
        self.prev_im = frames[2]
        w = frame.shape[1]
        h = frame.shape[0]
        q_frame = QImage(frame, w, h, 3 * w, QImage.Format_BGR888)
        self.prev_frame.setPixmap(QPixmap(q_frame))

    def update_plots(self):
        hot_marks, split_marks = self.control.get_data('thermal')
        if hot_marks is not None:
            self.mark_plotter.clear()
            self.mark_plotter.plot(hot_marks[:, 1], hot_marks[:, 2], s=0.05,
                                   plot_style='scatter', c=hot_marks[:, 0],
                                   marker='.', vmin=0, vmax=self.control.get_data('length'))

        self.raster_plotter.clear()
        for ind, sm in enumerate(split_marks):
            marks = np.unique(sm[:, 0])
            self.raster_plotter.plot(np.vstack((marks, marks)), [ind, ind + 1], c=MOUSE_COLORS_MPL[ind])
        self.raster_plotter.plot([self.control.frame_num, self.control.frame_num], [0, 2], 'k--')
        self.raster_plotter.gca().set_xlim(self.control.frame_num - 400, self.control.frame_num + 400)

        self.xy_plotter.clear()
        mice_xys = self.control.get_data('optical', time_win=12000)

        for ind, xy in enumerate(mice_xys):
            self.xy_plotter.plot(xy[0], xy[1], c=MOUSE_COLORS_MPL[ind])

        self.vel_plotter.clear()
        self.vel_plotter.plot([self.control.frame_num, self.control.frame_num], [0, 2], 'k--')
        self.vel_plotter.gca().set_xlim(self.control.frame_num - 400, self.control.frame_num + 400)

class SlideInputer(QGroupBox):
    def __init__(self, name, label=None):
        if label is None:
            super().__init__(name)
        else:
            super().__init__(label)
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
    settings_dict = {}
    def __init__(self, name, arena_type='circle'):
        super().__init__(name)
        self.arena_type = arena_type
        self.custom_pt_num = 3
        self.layout = QGridLayout()
        arena_ids = ['circle', 'rectangle', 'custom']
        default_sets = [(0, 0, 0), (0, 0, 0, 0), []]
        for a, s in zip(arena_ids, default_sets):
            self.settings_dict[a] = s
        list_wid = QComboBox()
        list_wid.addItems(arena_ids)
        list_wid.setCurrentText(arena_type)
        list_wid.currentTextChanged.connect(self.update_controls)
        self.update_controls(self.arena_type)
        self.layout.addWidget(list_wid, 0, 0, 1, 3)
        self.setLayout(self.layout)


    def update_controls(self, arena, data=None):
        for sc in self.sub_controls:
            sc.setParent(None)
        self.sub_controls = []
        self.arena_type = arena
        cur_settings = self.settings_dict[self.arena_type]
        if data is not None:
            cur_settings = data

        if arena == 'circle' or arena == 'rectangle':
            x_slide = QSlider(Qt.Orientation.Horizontal, minimum=1, maximum=1000, value=cur_settings[0])
            y_slide = QSlider(Qt.Orientation.Horizontal, minimum=1, maximum=1000, value=cur_settings[1])
            self.layout.addWidget(x_slide, 1, 1, 1, 1)
            self.layout.addWidget(y_slide, 2, 1, 1, 1)
            l0 = QLabel('Center X')
            self.layout.addWidget(l0, 1, 0, 1, 1)
            l1 = QLabel('Center Y')
            self.layout.addWidget(l1, 2, 0, 1, 1)
            if arena == 'circle':
                slide = QSlider(Qt.Orientation.Horizontal, minimum=1, maximum=1000, value=cur_settings[2])
                self.layout.addWidget(slide, 1, 3, 1, 1)
                l2 = QLabel('Radius')
                self.layout.addWidget(l2, 1, 2, 1, 1)
                for c in (x_slide, y_slide, slide, l0, l1, l2):
                    self.sub_controls.append(c)

            if arena == 'rectangle':
                vals = cur_settings[2:]
                slides = []
                for i, v in zip(range(2), vals):
                    slide = QSlider(Qt.Orientation.Horizontal, minimum=1, maximum=1000, value=v)
                    slides.append(slide)
                    self.layout.addWidget(slide, i+1, 3, 1, 1)

                l2 = QLabel('Width')
                l3 = QLabel('Height')
                for i, l in zip(((1, 2), (2, 2)), (l2, l3)):
                    self.layout.addWidget(l, i[0], i[1], 1, 1)
                for c in (x_slide, y_slide, slides[0], slides[1], l0, l1, l2, l3):
                    self.sub_controls.append(c)


        if arena == 'custom':
            draw_but = QPushButton('Draw Arena')
            draw_but.clicked.connect(self.draw_arena)
            self.sub_controls.append(draw_but)
            self.layout.addWidget(draw_but, 2, 1, 1, 1)
            pt_box = QLineEdit(str(self.custom_pt_num))
            pt_box.setValidator(QIntValidator(3, 15))
            def update_pt_num(text):
                self.custom_pt_num = text
            pt_box.textChanged.connect(update_pt_num)
            self.layout.addWidget(pt_box, 1, 1, 1, 1)
            lab = QLabel('Number of Points')
            self.layout.addWidget(lab, 1, 0, 1, 1)
            self.sub_controls.append(pt_box)
        for knob in self.sub_controls:
            if type(knob) is QSlider:
                knob.valueChanged.connect(self.update_settings)
        self.update_settings()

    def get_arena_data(self):
        outs_args = []
        if self.arena_type != 'custom':
            for sc in self.sub_controls:
                if isinstance(sc, QSlider):
                    outs_args.append(sc.value())
        else:
            outs_args = self.custom_pts
        return self.arena_type, outs_args

    def update_settings(self):
        self.settings_dict[self.arena_type] = self.get_arena_data()[1]

    def draw_arena(self):
        f = plt.figure(label='Custom Arena GUI')
        plt.imshow(self.frame_buf)
        plt.title(f'Click {self.custom_pt_num} times to define arena')
        pnts = plt.ginput(n=int(self.custom_pt_num), timeout=0)
        plt.close(f)
        self.custom_pts = np.array(pnts).astype(int)

    def set_frame(self, frame):
        self.frame_buf = frame

    def set_values(self, data):
        a_type = data[1]
        a_d = (data[0][0], data[0][1], data[2])
        if a_type == 'rectangle':
            a_d = (data[0][0], data[0][1], data[2][0], data[2][1])
        self.update_controls(a_type, data=a_d)

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
        self.cmap = 'summer'
        self.current_pobj = []
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

    def clear(self):
        if len(self.current_pobj) > 0:
            for pobj_list in self.current_pobj:
                if type(pobj_list) is PathCollection:
                    pobj_list.remove()
                else:
                    for pobj in pobj_list:
                        pobj.remove()
            self.current_pobj = []

    def plot(self, *args, plot_style=None, **kwargs):
        my_ax = self.gca()
        if plot_style is not None:
            if plot_style == 'scatter':
                self.current_pobj.append(my_ax.scatter(args[0], args[1], cmap=self.cmap, **kwargs))
        else:
            self.current_pobj.append(my_ax.plot(*args, **kwargs))
        # self.canvas.draw()

    def draw(self):
        self.canvas.draw()

    def colorbar(self, min_val, max_val, label='', orient='vertical'):
        norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)

        self.canvas.fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=self.cmap),
                     ax=self.gca(), orientation=orient, label=label)


class PtectApp:
    def __init__(self, data_folder: str = None):
        app = QApplication(sys.argv)
        gui = PtectMainWindow(data_folder=data_folder)
        sys.exit(app.exec())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = PtectMainWindow()
    sys.exit(app.exec())
