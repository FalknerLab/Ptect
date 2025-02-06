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
from territorytools.urine import Peetector, PtectPipe, split_urine_data, proj_urine_across_time
from territorytools.ttclasses import MDcontroller
from territorytools.plotting import add_territory_circle, territory_heatmap
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
    """
    Opens a file dialog to select a directory.

    Returns
    -------
    str
        Path to the selected directory.
    """
    dialog = QFileDialog()
    data_fold = dialog.getExistingDirectory()
    return data_fold

def get_save_dialog(filter='', suffix=''):
    """
    Opens a file dialog to select a save file path.

    Parameters
    ----------
    filter : str, optional
        Filter for the file dialog (default is '').
    suffix : str, optional
        Suffix to add to the file name (default is '').

    Returns
    -------
    str or None
        Path to the selected save file or None if no file was selected.
    """
    dialog = QFileDialog()
    save_path = dialog.getSaveFileName(filter=filter)
    if save_path[1] != '':
        out_path = save_path[0] + suffix + save_path[1]
        return out_path
    else:
        return None


class PtectController:
    data_buf = []
    hot_data = np.empty((0, 3))
    cool_data = np.empty((0, 2))
    def __init__(self, data_folder: str = None):
        """
        Initializes the PtectController.

        Parameters
        ----------
        data_folder : str, optional
            Path to the data folder (default is None).
        """
        if data_folder is None:
            data_folder = get_data_dialog()
        self.data_folder = data_folder
        self.test = 0
        self.frame_num = 0
        self.last_frame = 0
        self.t_frame = 0
        self.control_hz = 0
        self.save_path = None
        self.show_steps = False
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
            sleap_file = h5py.File(t_files['top.h5'], 'r')
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
            self.total_frames = int(self.optical_vid.get(cv2.CAP_PROP_FRAME_COUNT))

    def set_frame(self, frame_num: int):
        """
        Sets the current frame number.

        Parameters
        ----------
        frame_num : int
            Frame number to set.
        """
        if frame_num > self.optical_vid.get(cv2.CAP_PROP_FRAME_COUNT)-1:
            frame_num = 0
        self.last_frame = self.frame_num
        self.frame_num = frame_num
        self.ptect.set_frame(frame_num)

    def get_ptect_data(self):
        return (self.metadata.get_val('Territory/ptect_smooth_kern'),
                self.metadata.get_val('Territory/ptect_heat_thresh'),
                self.metadata.get_val('Territory/ptect_dilate_kern'),
                self.metadata.get_val('Territory/ptect_cool_thresh'),
                self.metadata.get_val('Territory/ptect_time_thresh'))

    def get_data(self, which_data, time_win=None):
        """
        Retrieves data based on the specified type.

        Parameters
        ----------
        which_data : str
            Type of data to retrieve.
        time_win : int, optional
            Time window for data retrieval (default is None).

        Returns
        -------
        various
            Retrieved data based on the specified type.
        """
        start_ind = self.frame_num - 1
        if time_win is not None:
            start_ind = self.frame_num - time_win
            start_ind = max(start_ind, 0)

        if which_data == 'therm_len':
            return self.ptect.get_length()

        if which_data == 'length':
            return self.total_frames

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

    def clear_data(self):
        self.hot_data = np.empty((0, 3))

    def get_optical_frame(self):
        """
        Retrieves the current optical frame.

        Returns
        -------
        tuple
            A tuple containing the return status and the frame.
        """
        op_frame_num = round(self.frame_num * (self.op_hz / self.control_hz))
        op_frame_num = min(op_frame_num, self.optical_vid.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        op_frame_num = int(op_frame_num)
        self.optical_vid.set(cv2.CAP_PROP_POS_FRAMES, op_frame_num)
        # if op_frame_num != self.frame_num:
        #     self.optical_vid.set(cv2.CAP_PROP_POS_FRAMES, op_frame_num)
        # if self.frame_num - self.last_frame > 1:
        #     self.optical_vid.set(cv2.CAP_PROP_POS_FRAMES, op_frame_num)
        # elif self.frame_num == 0:
        #     self.optical_vid.set(cv2.CAP_PROP_FRAME_COUNT, 0)
        ret, frame = self.optical_vid.read()
        for ind, sd in enumerate(self.sleap_data):
            not_nan = ~np.isnan(sd[0, :, op_frame_num])
            good_pts = sd[:, not_nan, op_frame_num].astype(int)
            for x, y in zip(good_pts[0, :], good_pts[1, :]):
                cv2.circle(frame, (x, y), 5, MOUSE_COLORS_BGR[ind], -1)
        return ret, frame

    def read_next_frame(self, *args):
        """
        Reads the next frame and updates the data.

        Parameters
        ----------
        *args : tuple
            Additional arguments for resizing the frame.

        Returns
        -------
        tuple
            A tuple containing the combined image, optical frame, and thermal frame.
        """
        resize_w, resize_h = 1280, 480
        if len(args) == 2:
            resize_w, resize_h = args[:2]

        c_im = np.empty((resize_h, resize_w, 3))
        ret, frame = self.get_optical_frame()
        if ret:
            rs_raw = cv2.resize(frame, (resize_w // 2, resize_h))
        else:
            rs_raw = np.zeros((resize_h, resize_w // 2, 3))
        out_frame = None
        last_frame = self.t_frame
        new_frame = round(self.frame_num * (self.therm_hz / self.control_hz)) + self.t_offset_frames
        # if new_frame > last_frame:/
        urine_data, out_frame, f_num = self.ptect.peetect_next_frame(return_frame=True)
        rs_urine = cv2.resize(out_frame, (resize_w // 2, resize_h))
        self.update_data(urine_data)
        self.t_frame = new_frame
        self.last_t_frame = out_frame

        rs_urine = cv2.resize(self.last_t_frame, (resize_w // 2, resize_h))

        if self.show_steps:
            c_im = cv2.resize(self.last_t_frame, (resize_w, resize_h))
        else:
            c_im = np.hstack((rs_raw, rs_urine))

        self.set_frame(self.frame_num + 1)
        return c_im.astype('uint8'), frame, out_frame

    def update_data(self, data):
        """
        Updates the data buffer with new data.

        Parameters
        ----------
        data : list
            New data to update.
        """
        if len(data[0]) > 0:
            this_d = data[0]
            this_d[:, 0] = self.frame_num
            new_marks, new_inds = intersect2d(self.hot_data[:, 1:], this_d[:, 1:], return_int=False)
            self.hot_data = np.vstack((self.hot_data, this_d[new_inds, :]))

    def set_param(self, param, value):
        """
        Sets a parameter for the Peetector.

        Parameters
        ----------
        param : str
            Parameter name.
        value : various
            Value to set for the parameter.
        """
        match param:
            case 'dilate':
                self.ptect.dilate_kern = value
                self.metadata.set_key_val('Territory/ptect_dilate_kern', value)
            case 'smooth':
                self.ptect.smooth_kern = value
                self.metadata.set_key_val('Territory/ptect_smooth_kern', value)
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
            case 'show_steps':
                self.show_steps = value
                if self.show_steps:
                    self.ptect.frame_type = 2
                else:
                    self.ptect.frame_type = 1
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
        """
        Retrieves metadata based on the specified key.

        Parameters
        ----------
        md : str
            Metadata key.

        Returns
        -------
        various
            Retrieved metadata value.
        """
        match md:
            case 'arena':
                return self.metadata.get_val('Territory/thermal_center'), self.metadata.get_val('Territory/arena_type'), self.metadata.get_val('Territory/arena_data')
            case 'deadzone':
                return self.metadata.get_val('Territory/deadzone')

    def get_info(self):
        """
        Retrieves the metadata information as a string.

        Returns
        -------
        str
            Metadata information.
        """
        return str(self.metadata)

    def save_info(self):
        """
        Saves the metadata information.
        """
        self.metadata.save_metadata(self.metadata.file_name)

    def run_and_save(self, ppipe):
        """
        Runs the Peetector and saves the results.

        Parameters
        ----------
        ppipe : PtectPipe
            Pipe for the Peetector.
        """
        self.ptect.run_ptect(pipe=ppipe, save_path=self.save_path)

    def get_file_list(self):
        """
        Retrieves the list of territory files.

        Returns
        -------
        dict
            Dictionary containing the territory files.
        """
        return find_territory_files(self.data_folder)

    def load_output(self):
        """
        Loads the processed data.

        Returns
        -------
        various
            Processed data.
        """
        return process_all_data(self.data_folder)


class PtectGUIpipe(PtectPipe):
    def __init__(self, pipe: pyqtSignal(tuple)):
        """
        Initializes the PtectGUIpipe.

        Parameters
        ----------
        pipe : pyqtSignal
            Signal for the pipe.
        """
        self.pipe = pipe

    def send(self, *args):
        """
        Sends data through the pipe.

        Parameters
        ----------
        *args : tuple
            Data to send.
        """
        self.pipe.emit(args[0])

    def set_buffer(self, data):
        """
        Sets the buffer data.

        Parameters
        ----------
        data : various
            Data to set in the buffer.
        """
        self.buffer = data


class RunSignals(QObject):
    progress = pyqtSignal(tuple)
    finished = pyqtSignal()


class PtectRunner(QObject):
    def __init__(self, ptect_cont: PtectController):
        """
        Initializes the PtectRunner.

        Parameters
        ----------
        ptect_cont : PtectController
            Controller for the Peetector.
        """
        super().__init__()

        self.signals = RunSignals()
        self.ptect_cont = ptect_cont
        self.gui_pipe = PtectGUIpipe(self.signals.progress)

    @pyqtSlot()
    def run(self):
        """
        Runs the Peetector and emits the finished signal.
        """
        self.ptect_cont.run_and_save(self.gui_pipe)
        self.signals.finished.emit()

    def set_stop_cb(self, stop_cb):
        """
        Sets the stop callback.

        Parameters
        ----------
        stop_cb : callable
            Callback function to call when the process is stopped.
        """
        self.signals.finished.connect(stop_cb)


class PtectThread(QThread):
    result = pyqtSignal(tuple)
    started = pyqtSignal()

    def __init__(self, ptect_cont, parent=None):
        """
        Initializes the PtectThread.

        Parameters
        ----------
        ptect_cont : PtectController
            Controller for the Peetector.
        parent : QObject, optional
            Parent object (default is None).
        """
        super().__init__(parent)
        self.worker = PtectRunner(ptect_cont)
        self.started.connect(self.worker.run)
        self.worker.signals.progress.connect(self.emit_worker_output)

    def spawn_workers(self, stop_cb=None):
        """
        Spawns worker threads.

        Parameters
        ----------
        stop_cb : callable, optional
            Callback function to call when the process is stopped (default is None).
        """
        self.worker.gui_pipe.set_buffer(False)
        if stop_cb is not None:
            self.worker.set_stop_cb(stop_cb)
        self.worker.moveToThread(self)
        self.start()
        self.started.emit()

    def emit_worker_output(self, output):
        """
        Emits the worker output.

        Parameters
        ----------
        output : tuple
            Output data from the worker.
        """
        self.result.emit(output)

    def kill(self):
        """
        Kills the worker thread.
        """
        self.worker.gui_pipe.set_buffer(True)


class PtectWindow(QWidget):
    def __init__(self, ptect_cont: PtectController=None, parent=None):
        """
        Initializes a generic PtectWindow with logo.

        Parameters
        ----------
        ptect_cont : PtectController, optional
            Controller for the Peetector (default is None).
        parent : QWidget, optional
            Parent widget (default is None).
        """
        super().__init__()
        self.parent = parent
        self.control = ptect_cont
        icon_path = os.path.abspath('../resources/ptect_icon.png')
        self.icon = QIcon(icon_path)
        self.setWindowIcon(self.icon)


class PtectMainWindow(QMainWindow):
    def __init__(self, data_folder=None):
        """
        Initializes the PtectMainWindow, which manages each subwindow

        Parameters
        ----------
        data_folder : str, optional
            Path to the data folder (default is None).
        """
        super().__init__()
        print('Importing Data...')
        self.control = PtectController(data_folder=data_folder)
        self.preview = PtectPreviewWindow(ptect_cont=self.control, parent=self)
        self.run_win = PtectRunWindow(self.control, parent=self)
        self.data_win = PtectDataWindow(self.control, parent=self)
        self.er_win = None
        self.preview.show()

    def start_ptect(self):
        """
        Starts the Peetector process and switch to run window.
        """
        self.preview.hide()
        self.run_win.show()
        save_path = get_save_dialog('.npz', '_ptect')
        if save_path is not None:
            self.control.save_path = save_path
            self.run_win.thread_pool.spawn_workers(stop_cb=self.stop_ptect)
        else:
            self.preview.show()

    def stop_ptect(self):
        """
        Callback for when Ptect stops, switch back to preview window.
        """
        self.run_win.hide()
        self.preview.show()

    def no_out(self):
        """
        Displays an error window when no output data is found.
        """
        self.er_win = PtectWindow(parent=self)
        data_f = self.control.data_folder
        text = QLabel()
        text.setText(f'No output data found in: {data_f}\nUse "Run and Save" to generate _ptect.npz file.\nData folder must contain _ptect.npz file')
        er_layout = QHBoxLayout()
        er_layout.addWidget(text)
        self.er_win.setLayout(er_layout)
        self.er_win.show()

    def show_output(self):
        """
        Displays the output data in the data window.
        """
        data_files = self.control.get_file_list()
        ptect_npz = data_files['ptect.npz']
        if ptect_npz is None:
            self.no_out()
        else:
            self.preview.hide()
            self.data_win.init_plots()
            self.data_win.show()

    def close_output(self):
        """
        Closes the data window and switches back to the preview window.
        """
        self.data_win.hide()
        self.preview.show()


class PtectDataWindow(PtectWindow):
    def __init__(self, ptect_cont, parent=None):
        """
        Initializes the PtectDataWindow.

        Parameters
        ----------
        ptect_cont : PtectController
            Controller for the Peetector.
        parent : QWidget, optional
            Parent widget (default is None).
        """
        super().__init__(ptect_cont, parent)
        self.resize(640, 480)
        self.setWindowTitle('Output from: ' + self.control.data_folder)
        self.grid = QGridLayout(self)
        self.plot_dict = self.init_plots()
        self.setLayout(self.grid)

    def init_plots(self):
        """
        Initializes the plots for the data window.

        Returns
        -------
        dict
            Dictionary containing the plot widgets.
        """
        plot_dict = {}
        data_files = self.control.get_file_list()
        ptect_npz = data_files['ptect.npz']
        if ptect_npz is not None:
            data = self.control.load_output()
            xy_plot = PlotWidget(self)
            mark_plot = PlotWidget(self)
            vel_plot = PlotWidget(self)
            rast_plot = PlotWidget(self)
            for p in (xy_plot, mark_plot):
                ax = p.gca()
                add_territory_circle(ax, facecolor='w')
                ax.set_xlim(-32, 32)
                ax.set_ylim(-32, 32)

            for i, m in enumerate(data):
                xy_plot.plot(m['x_cm'], m['y_cm'])
                vel_plot.plot(m['velocity'], color=MOUSE_COLORS_MPL[i], alpha=1/len(data))
                marks = m['urine_data']
                hot_marks, cool_marks = split_urine_data(marks)
                hot_proj, times_h = proj_urine_across_time(hot_marks)
                cool_proj, times_c = proj_urine_across_time(cool_marks)
                mark_plot.plot(cool_proj[:, 0], cool_proj[:, 1], plot_style='scatter',
                               cmap='grey', c=times_c, edgecolor='b', alpha=0.5)
                mark_plot.plot(hot_proj[:, 0], hot_proj[:, 1], plot_style='scatter',
                               cmap='grey', c=times_h, edgecolor='r', alpha=0.5)
                # h_rast, c_rast = make_mark_raster(marks)
                rast_plot.plot(times_h, np.ones_like(times_h)*i, plot_style='scatter', color=MOUSE_COLORS_MPL[i],
                               edgecolor='r')
                rast_plot.plot(times_c, np.ones_like(times_c) * i - 0.1, plot_style='scatter', color=MOUSE_COLORS_MPL[i],
                               edgecolor='b')

                hm_plot = PlotWidget(self)
                hm_ax = hm_plot.gca()
                territory_heatmap(m['x_cm'], m['y_cm'], ax=hm_ax)
                self.grid.addWidget(hm_plot, 1, i, 1, 1)

            rast_plot.gca().set_ylim(-0.25, 1.25)
            x_ticks = np.arange(len(data)).astype(int)
            names = np.array(['Mouse0', 'Mouse1'])
            rast_plot.gca().set_yticks(x_ticks)
            rast_plot.gca().set_yticklabels(names[x_ticks])

            plot_dict['xy'] = xy_plot
            plot_dict['marks'] = mark_plot
            self.grid.addWidget(xy_plot, 0, 0, 1, 1)
            self.grid.addWidget(mark_plot, 0, 1, 1, 1)
            self.grid.addWidget(vel_plot, 2, 0, 1, 2)
            self.grid.addWidget(rast_plot, 3, 0, 1, 2)
        return plot_dict

    def closeEvent(self, event):
        """
        Handles the close event for the data window.

        Parameters
        ----------
        event : QCloseEvent
            Close event.
        """
        self.parent.close_output()


class PtectRunWindow(PtectWindow):
    def __init__(self, ptect_cont, parent=None):
        """
        Initializes the PtectRunWindow.

        Parameters
        ----------
        ptect_cont : PtectController
            Controller for the Peetector.
        parent : QWidget, optional
            Parent widget (default is None).
        """
        super().__init__(ptect_cont, parent)

        res = self.screen().size()
        self.setGeometry(int(res.width()/2 - 60), int(res.height()/2 - 210), 420, 120)
        self.setFixedSize(420, 120)
        self.setWindowTitle('Running Ptect...')

        finish_i_path = os.path.abspath('../resources/finish_icon.png')
        self.finish_w = QLabel(self)
        self.finish_w.setPixmap(QPixmap(finish_i_path))
        self.finish_w.move(340, 60)

        self.message = QLabel('Ptecting... On Frame:', self)
        self.message.setGeometry(10, 10, 400, 30)
        self.mouse_i_path = os.path.abspath('../resources/mouse_icon.png')
        self.icon_w = QLabel(self)
        self.icon_w.setPixmap(QPixmap(self.mouse_i_path))
        self.icon_w.move(20, 55)

        self.end_x = self.finish_w.pos().x()

        self.start = 0
        self.stop = 0
        self.thread_pool = PtectThread(self.control, self)

        self.register_actions()

    def register_actions(self):
        """
        Registers actions for the run window.
        """
        self.thread_pool.result.connect(self.update_run)

    def update_run(self, s):
        """
        Updates the run progress.

        Parameters
        ----------
        s : tuple
            Tuple containing the current and total frame numbers.
        """
        frac_done = s[0] / s[1]
        next_x = int(frac_done*self.end_x)
        orig_p = self.icon_w.pos()
        self.icon_w.move(next_x, orig_p.y())
        self.icon_w.update()
        new_txt = 'Ptecting... On Frame: ' + str(s[0]) + ' of ' + str(s[1])
        self.message.setText(new_txt)
        self.message.update()

    def closeEvent(self, event):
        """
        Handles the close event for the run window.

        Parameters
        ----------
        event : QCloseEvent
            Close event.
        """
        self.thread_pool.kill()
        self.parent.stop_ptect()


class PtectPreviewWindow(PtectWindow):
    playing = False
    thresh_controls = []

    def __init__(self, ptect_cont, parent=None):
        """
        Initializes the PtectPreviewWindow.

        Parameters
        ----------
        ptect_cont : PtectController
            Controller for the Peetector.
        parent : QWidget, optional
            Parent widget (default is None).
        """
        super().__init__(ptect_cont, parent)
        res = self.screen().size()
        self.resize(int(res.width()*0.8), int(res.height()*0.8))
        self.setWindowTitle('Ptect Preview GUI')

        self.preview_frame_w = 1960
        self.preview_frame_h = 840

        self.control = ptect_cont

        self.layout = QGridLayout()
        self.prev_frame = QLabel()

        self.add_controls()
        init_params = self.control.get_ptect_data()
        self.set_controls(init_params)

        print('Initializing GUI...')
        first_frames = self.control.read_next_frame(self.preview_frame_w, self.preview_frame_h)
        self.prev_im = first_frames[2]

        q_im = QImage(first_frames[0],
                         self.preview_frame_w,
                         self.preview_frame_h,
                         3 * self.preview_frame_w,
                         QImage.Format_BGR888)
        self.prev_frame.setPixmap(QPixmap(q_im))
        self.prev_frame.setScaledContents(True)
        self.layout.addWidget(self.prev_frame, 0, 0, 3, 2)



        self.setLayout(self.layout)
        self.prev_frame.show()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(10)

        self.draw_timer = QTimer(self)
        self.draw_timer.timeout.connect(self.draw_plots)
        self.draw_timer.start(1000)

    def add_controls(self):
        """
        Adds control widgets to the preview window.
        """
        params = ['smooth', 'heat_thresh', 'dilate', 'cool_thresh', 'time_thresh']
        disp_names = ['Smooth Size', 'Heat Thesh', 'Dilate/Erode Size', 'Cool Thresh', 'Check # of Frames']
        mins = [1, 0, 1, 0, 0]
        maxs = [15, 255, 15, 255, 100]

        num_slides = len(params)
        end_slides = 2+num_slides
        for i, p, n, mn, mx in zip(range(2, end_slides), params, disp_names, mins, maxs):
            slider = SlideInputer(p, label=n, low=mn, high=mx)
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
            conts = self.control.get_ptect_data()
            self.set_controls(slide_settings=conts)
        load_but.clicked.connect(load_data)

        run_but = QPushButton('Run and Save')
        def run_ptect():
            self.parent.start_ptect()
        run_but.clicked.connect(run_ptect)

        show_but = QPushButton('Show Output')
        def show_data():
            self.parent.show_output()
        show_but.clicked.connect(show_data)

        but_group = QWidget()
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(play)
        sub_layout.addWidget(load_but)
        sub_layout.addWidget(run_but)
        sub_layout.addWidget(show_but)
        but_group.setLayout(sub_layout)
        self.layout.addWidget(but_group, 5, 0, 1, 2)


        info_gb = QGroupBox('Run Info')
        info_box = QVBoxLayout()
        scroller = QScrollArea()
        self.run_info = QLabel(self.control.get_info())
        scroller.setWidget(self.run_info)
        info_box.addWidget(scroller)
        info_gb.setLayout(info_box)
        self.layout.addWidget(info_gb, 2, 7, 1, 1)

        set_frame = QCheckBox('Show Steps')
        def set_frame_type():
            if set_frame.isChecked():
                self.control.set_param('show_steps', True)
            else:
                self.control.set_param('show_steps', False)
        set_frame.clicked.connect(set_frame_type)
        self.layout.addWidget(set_frame, 1, 2, 1, 1)

        dz_but = QPushButton('Add Dead Zone')
        dz_but.clicked.connect(self.set_dz)
        self.layout.addWidget(dz_but, 1, 3, 1, 1)

        self.dz_pt_box = QLineEdit()
        self.dz_pt_box.setValidator(QIntValidator(3, 15))
        pt_lab = QLabel('# of Points')
        self.layout.addWidget(self.dz_pt_box, 1, 4, 1, 1)
        self.layout.addWidget(pt_lab, 1, 5, 1, 1)

        clear_dz_but = QPushButton('Clear Dead Zones')
        def clear_dz():
            self.control.set_param('deadzone', [])
        clear_dz_but.clicked.connect(clear_dz)
        self.layout.addWidget(clear_dz_but, 1, 6, 1, 1)

        save_info_but = QPushButton('Save Info')
        save_info_but.clicked.connect(self.control.save_info)
        self.layout.addWidget(save_info_but, 1, 7, 1, 1)


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
        labels = ['Self', 'Other']
        for i in range(len(vels)):
            vel_ax.plot(i + vels[i]/max(vels[i]), c=MOUSE_COLORS_MPL[i], label=labels[i])
        vel_ax.legend()
        self.layout.addWidget(self.vel_plotter, 3, 2, 1, 6)

        self.raster_plotter = PlotWidget()
        rast_ax = self.raster_plotter.gca()
        rast_ax.set_xlim(0, self.control.get_data('length'))
        rast_ax.set_title('Marking Raster')
        self.layout.addWidget(self.raster_plotter, 4, 2, 1, 6)

        self.reset_marks_but = QPushButton('Reset Marks')
        def reset_marks():
            self.mark_plotter.clear()
            self.control.clear_data()
        self.reset_marks_but.clicked.connect(reset_marks)
        self.layout.addWidget(self.reset_marks_but, 3, 1, 1, 1, Qt.AlignLeft)

        self.arena_controls = ArenaSelector('Arena Controls')
        self.layout.addWidget(self.arena_controls, 0, 2, 1, num_slides+3)

        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setMinimum(1)
        self.scrubber.setMaximum(self.control.get_data('length'))
        def change_frame(val):
            self.control.set_frame(val)
        self.scrubber.valueChanged.connect(change_frame)
        self.layout.addWidget(self.scrubber, 6, 0, 4, 1)

        self.frame_counter = QLabel('Frame: ')
        self.layout.addWidget(self.frame_counter, 6, 1, 1, 1)


    def set_controls(self, slide_settings=None):
        """
        Sets the widget values for the preview window.
        """
        arena_data = self.control.get_metadata('arena')
        dz = self.control.get_metadata('deadzone')
        self.control.set_param('deadzone', dz)
        self.arena_controls.set_values(arena_data)
        self.control.set_param('arena', self.arena_controls.get_arena_data())
        self.arena_controls.set_frame(self.control.test_frame)
        if slide_settings is not None:
            for s, val in zip(self.thresh_controls, slide_settings):
                s.set_value(val)
                self.control.set_param(s.id, val)


    def set_dz(self):
        """
        Sets the arena dead zone parameter.
        """
        self.control.set_param('deadzone', self.dz_pt_box.text())

    def update_gui(self):
        """
        Updates the GUI elements at each step of the main QTimer.
        """
        for c in self.thresh_controls:
            param, val = c.get_value()
            self.control.set_param(param, val)

        self.run_info.setText(self.control.get_info())

        arena_params = self.arena_controls.get_arena_data()
        self.control.set_param('arena', arena_params)
        if self.playing:
            self.update_plots()
            self.update_video()

        f = self.control.frame_num
        tot_f = self.control.get_data('length')
        self.frame_counter.setText(f'Frame: {f} of {tot_f}')
        self.scrubber.setValue(f)

    def draw_plots(self):
        """
        Draws the plots in the preview window.
        """
        for p in (self.xy_plotter, self.raster_plotter, self.vel_plotter, self.mark_plotter):
            p.draw()

    def update_video(self):
        """
        Updates the video frame in the preview window.
        """
        frames = self.control.read_next_frame(self.preview_frame_w, self.preview_frame_h)
        frame = frames[0]
        self.prev_im = frames[2]
        w = frame.shape[1]
        h = frame.shape[0]
        q_frame = QImage(frame, w, h, 3 * w, QImage.Format_BGR888)
        self.prev_frame.setPixmap(QPixmap(q_frame))

    def update_plots(self):
        """
        Updates the plots in the preview window.
        """
        hot_marks, split_marks = self.control.get_data('thermal')
        if hot_marks is not None:
            self.mark_plotter.clear()
            self.mark_plotter.plot(hot_marks[:, 1], hot_marks[:, 2], s=0.05,
                                   plot_style='scatter', c=hot_marks[:, 0], cmap='summer',
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
    def __init__(self, name, label=None, low=0, high=255):
        """
        Initializes the SlideInputer.

        Parameters
        ----------
        name : str
            Name of the slider.
        label : str, optional
            Label for the slider (default is None).
        """
        if label is None:
            super().__init__(name)
        else:
            super().__init__(label)
        self.id = name
        slide_group = QVBoxLayout()
        self.slide = QSlider()
        self.slide.setMinimum(low)
        self.slide.setMaximum(high)
        self.slide.valueChanged.connect(self.update_ebox)

        self.ebox = QLineEdit()
        self.ebox.setValidator(QIntValidator())
        self.ebox.setMaxLength(3)
        self.ebox.textChanged.connect(self.update_slide)

        for w in (self.slide, self.ebox):
            slide_group.addWidget(w)
        self.setLayout(slide_group)

    def update_slide(self, val):
        """
        Updates the slider value.

        Parameters
        ----------
        val : str
            Value to set for the slider.
        """
        if len(val) > 0:
            val = int(val)
            self.slide.setValue(val)

    def update_ebox(self, val):
        """
        Updates the edit box value.

        Parameters
        ----------
        val : int
            Value to set for the edit box.
        """
        val = str(val)
        self.ebox.setText(val)

    def get_value(self):
        """
        Retrieves the current value of the slider.

        Returns
        -------
        tuple
            Tuple containing the slider ID and value.
        """
        return self.id, self.slide.value()

    def set_value(self, value):
        self.slide.setValue(value)


class ArenaSelector(QGroupBox):
    size = 0
    sub_controls = []
    frame_buf = None
    custom_pts = []
    settings_dict = {}
    def __init__(self, name, arena_type='circle'):
        """
        Initializes the ArenaSelector.

        Parameters
        ----------
        name : str
            Name of the arena selector.
        arena_type : str, optional
            Type of the arena (default is 'circle').
        """
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
        """
        Updates the controls for the arena selector.

        Parameters
        ----------
        arena : str
            Type of the arena.
        data : various, optional
            Data for the arena (default is None).
        """
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
            self.sub_controls.append(lab)
        for knob in self.sub_controls:
            if type(knob) is QSlider:
                knob.valueChanged.connect(self.update_settings)
        self.update_settings()

    def get_arena_data(self):
        """
        Retrieves the arena data.

        Returns
        -------
        tuple
            Tuple containing the arena type and data.
        """
        outs_args = []
        if self.arena_type != 'custom':
            for sc in self.sub_controls:
                if isinstance(sc, QSlider):
                    outs_args.append(sc.value())
        else:
            outs_args = self.custom_pts
        return self.arena_type, outs_args

    def update_settings(self):
        """
        Updates the settings for the arena selector.
        """
        self.settings_dict[self.arena_type] = self.get_arena_data()[1]

    def draw_arena(self):
        """
        Draws the custom arena.
        """
        f = plt.figure(label='Custom Arena GUI')
        plt.imshow(self.frame_buf)
        plt.title(f'Click {self.custom_pt_num} times to define arena')
        pnts = plt.ginput(n=int(self.custom_pt_num), timeout=0)
        plt.close(f)
        self.custom_pts = np.array(pnts).astype(int)

    def set_frame(self, frame):
        """
        Sets the frame buffer.

        Parameters
        ----------
        frame : various
            Frame buffer to set.
        """
        self.frame_buf = frame

    def set_values(self, data):
        """
        Sets the values for the arena selector.

        Parameters
        ----------
        data : various
            Data to set.
        """
        a_type = data[1]
        a_d = (data[0][0], data[0][1], data[2])
        if a_type == 'rectangle':
            a_d = (data[0][0], data[0][1], data[2][0], data[2][1])
        self.update_controls(a_type, data=a_d)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        """
        Initializes an MplCanvas to display Matplotlib plots via QTAgg
        """
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111)
        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)


class PlotWidget(QWidget):
    def __init__(self, parent=None):
        """
        Initializes the PlotWidget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget (default is None).
        """
        QWidget.__init__(self, parent)
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
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        sm = matplotlib.cm.ScalarMappable(norm=norm)
        self.color_bar = self.canvas.fig.colorbar(sm, ax=self.gca())
        self.color_bar.ax.set_visible(False)

    def gca(self):
        """
        Retrieves the current axis from the MplCanvas

        Returns
        -------
        matplotlib.axes.Axes
            Current axis.
        """
        return self.canvas.ax

    def clear(self):
        """
        Clears the plot.
        """
        if len(self.current_pobj) > 0:
            for pobj_list in self.current_pobj:
                if type(pobj_list) is PathCollection:
                    pobj_list.remove()
                else:
                    for pobj in pobj_list:
                        pobj.remove()
            self.current_pobj = []

    def plot(self, *args, plot_style=None, **kwargs):
        """
        Plots the data.

        Parameters
        ----------
        *args : tuple
            Data to plot.
        plot_style : str, optional
            Style of the plot (default is None).
        **kwargs : dict
            Additional keyword arguments for the plot. Passed to matplotlib.plot
        """
        my_ax = self.gca()
        if plot_style is not None:
            if plot_style == 'scatter':
                self.current_pobj.append(my_ax.scatter(args[0], args[1], *args[2:], **kwargs))
        else:
            self.current_pobj.append(my_ax.plot(*args, **kwargs))
        # self.canvas.draw()

    def draw(self):
        """
        Draws the plot.
        """
        self.canvas.draw()

    def colorbar(self, min_val, max_val, label='', orient='vertical', cmap='summer'):
        """
        Adds a colorbar to the plot.

        Parameters
        ----------
        min_val : float
            Minimum value for the colorbar.
        max_val : float
            Maximum value for the colorbar.
        label : str, optional
            Label for the colorbar (default is '').
        orient : str, optional
            Orientation of the colorbar (default is 'vertical').
        cmap : str, optional
            Colormap for the colorbar (default is 'summer').
        """
        # for a in self.canvas.fig.axes:
        #     if a.label == 'colorbar':
        #         self.canvas.fig.axes.remove(a)
        # self.color_bar.remove()

        self.color_bar.ax.set_visible(True)
        norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        self.color_bar.update_normal(sm)
        self.color_bar.set_label(label)

        # self.color_bar = self.canvas.fig.colorbar(sm,
        #              ax=self.gca(), orientation=orient, label=label)


class PtectApp:
    def __init__(self, data_folder: str = None):
        """
        Initializes the PtectApp. Starts the QApplication process

        Parameters
        ----------
        data_folder : str, optional
            Path to the data folder (default is None).
        """
        self.app = QApplication(sys.argv)
        self.gui = PtectMainWindow(data_folder=data_folder)
        sys.exit(self.app.exec())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = PtectMainWindow()
    sys.exit(app.exec())
