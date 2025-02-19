import time
from abc import abstractmethod, ABC
from importlib import resources
from types import NoneType
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from PIL import ImageFont, ImageDraw, Image
from src.territorytools.utils import rotate_xy, intersect2d


FIRA_MONO = str(resources.files('resources').joinpath('assets').joinpath('fira_mono.ttf'))

def sleap_to_fill_pts(sleap_h5):
    """
    Converts SLEAP HDF5 data to fill points.

    Parameters
    ----------
    sleap_h5 : str
        Path to the SLEAP HDF5 file.

    Returns
    -------
    list of numpy.ndarray
        List of fill points for each frame.
    """
    with h5py.File(sleap_h5, "r") as f:
        locations = f["tracks"][:].T
    t, d1, d2, num_mice = np.shape(locations)
    fill_pts = []
    last_pts = [[None, None]]
    for i in range(t):
        move_pts = np.moveaxis(locations[i], 0, 2)
        all_xy = np.reshape(move_pts, (2, num_mice * d1))
        keep = ~np.all(np.isnan(all_xy), axis=0)
        if np.any(keep):
            k_pts = all_xy[:, keep].T
            last_pts = k_pts
        fill_pts.append(np.vstack(last_pts))
    return fill_pts


def expand_urine_data(urine_xys, times=None):
    """
    Expands urine data points with optional time information.

    Parameters
    ----------
    urine_xys : list of numpy.ndarray
        List of urine data points for each frame.
    times : list of float, optional
        List of time points corresponding to each frame (default is None).

    Returns
    -------
    numpy.ndarray
        Expanded urine data with optional time information.
    """
    num_urine_pnts_per_t = [len(xys) for xys in urine_xys]
    expanded_data = np.vstack(urine_xys)
    if times is not None:
        time_vec = np.zeros(expanded_data.shape[0])
        c = 0
        for ind, (t, nums) in enumerate(zip(times, num_urine_pnts_per_t)):
            time_vec[c:(c + nums)] = t
            c += nums
        expanded_data = np.hstack((time_vec[:, None], expanded_data))
    return expanded_data


def make_shape_mask(width, height, shape, cent_x, cent_y, *args):
    """
    Creates a mask of a specified shape.

    Parameters
    ----------
    width : int
        Width of the mask.
    height : int
        Height of the mask.
    shape : str
        Shape of the mask ('circle', 'rectangle', or 'polygon').
    cent_x : int
        X-coordinate of the center of the shape.
    cent_y : int
        Y-coordinate of the center of the shape.
    *args : tuple
        Additional arguments for the shape (e.g., radius for circle, width and height for rectangle, points for polygon).

    Returns
    -------
    numpy.ndarray
        Mask of the specified shape.
    """
    im = np.zeros((height, width))
    if shape == 'circle':
        radius = int(args[0])
        out_mask = cv2.circle(im, (cent_x, cent_y), radius, 255, -1)
    elif shape == 'rectangle':
        rect_w, rect_h = args[:]
        pt1 = (cent_x - (rect_w // 2), cent_y - (rect_h // 2))
        pt2 = (cent_x + (rect_w // 2), cent_y + (rect_h // 2))
        out_mask = cv2.rectangle(im, pt1, pt2, 255, -1)
    else:
        pts = []
        if len(args) > 0:
            pts = [np.array(args).astype(np.int32)]
        out_mask = cv2.fillPoly(im, pts, -1)
    out_mask = out_mask.astype('uint8')
    return out_mask

def map_xys(x0, y0, cent_x, cent_y, new_cent_x, new_cent_y, op_w, op_h, t_w, t_h):
    rel_xs = x0 - cent_x
    rel_ys = y0 - cent_y
    rel_prop_x = (rel_xs / op_w) * t_w
    rel_prop_y = (rel_ys / op_h) * t_h
    xs = new_cent_x + rel_prop_x
    ys = new_cent_y + rel_prop_y
    return xs, ys


class PtectPipe(ABC):
    """
    Abstract base class for a PtectPipe to pass data through and save in buffer.
    """
    buffer=[]
    @abstractmethod
    def send(self, *args):
        """
        Abstract method to define how data is sent through the pipe.

        Parameters
        ----------
        *args : tuple
            Data to send.
        """
        pass

    def read(self):
        """
        Reads data from the buffer.

        Returns
        -------
        list
            Data from the buffer.
        """
        return self.buffer

class Peetector:
    def __init__(self, avi_file, flood_pnts, dead_zones=[], cent_xy=(320, 212), px_per_cm=7.38188976378, check_frames=1,
                 hot_thresh=70, cold_thresh=30, s_kern=5, di_kern=5, hz=30, v_mask=None, frame_type=None, radius=30,
                 rot_ang=0, start_frame=0, optical_slp=None, use_op=False, op_center_x=707, op_center_y=541):
        """
        Initializes the Peetector.

        Parameters
        ----------
        avi_file : str
            Path to the AVI file.
        flood_pnts : str or list
            Path to the SLEAP file or list of flood points.
        dead_zones : list, optional
            List of dead zones (default is []).
        cent_xy : tuple, optional
            Center coordinates of the arena (default is (320, 212)).
        px_per_cm : float, optional
            Pixels per centimeter (default is 7.38188976378).
        check_frames : int, optional
            Number of frames to check for urine events (default is 1).
        hot_thresh : int, optional
            Threshold for hot events (default is 70).
        cold_thresh : int, optional
            Threshold for cold events (default is 30).
        s_kern : int, optional
            Smoothing kernel size (default is 5).
        di_kern : int, optional
            Dilation kernel size (default is 5).
        hz : int, optional
            Frame rate in Hz (default is 40).
        v_mask : numpy.ndarray, optional
            Valid zone mask (default is None).
        frame_type : int, optional
            Type of frame to return (default is None).
        radius : int, optional
            Radius of the arena (default is 30).
        rot_ang : float, optional
            Rotation angle in degrees (default is 0).
        start_frame : int, optional
            Starting frame number (default is 0).
        """
        self.thermal_vid = avi_file
        self.vid_obj = cv2.VideoCapture(avi_file)
        self.width = int(self.vid_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if type(flood_pnts) == str:
            print('Converting slp file to fill points...')
            self.fill_pts = sleap_to_fill_pts(flood_pnts)
        else:
            self.fill_pts = flood_pnts
        self.dead_zones = dead_zones
        self.arena_cnt = cent_xy
        self.arena_params = ()
        self.px_per_cm = px_per_cm
        self.heat_thresh = hot_thresh
        self.cool_thresh = cold_thresh
        self.time_thresh = check_frames
        self.smooth_kern = s_kern
        self.dilate_kern = di_kern
        self.hz = hz
        self.frame_type = frame_type
        self.rot_ang = rot_ang
        if v_mask is None:
            self.set_valid_arena('circle', int(radius * px_per_cm))
        else:
            self.valid_zone = v_mask
        self.total_frames = int(self.vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        self.buffer = PBuffer(check_frames)
        self.current_frame = start_frame
        self.set_frame(start_frame)
        self.output_buffer = ()
        self.arena_shape = 'circle'

        self.optical_slp = optical_slp
        self.op_fill_pts = None
        if optical_slp is not None:
            self.op_fill_pts = sleap_to_fill_pts(optical_slp)
            proj_pts = []
            for pts in self.op_fill_pts:
                if pts[0][0] is not None:
                    xs, ys = map_xys(pts[:, 0], pts[:, 1], op_center_x, op_center_y, self.arena_cnt[0], self.arena_cnt[1],
                                     self.width, self.height, self.arena_cnt[0], self.arena_cnt[1])
                    proj_pts.append(np.vstack((xs, ys)).T.astype(int))
                else:
                    proj_pts.append(pts)
            # pad_ar = np.empty((0, 2), dtype=np.ndarray)
            ds_op_inds = np.linspace(0, len(self.op_fill_pts), len(self.fill_pts) - start_frame).astype(np.int8)
            op_fill_pts = [proj_pts[i] for i in ds_op_inds]
            pad_op_fills = np.repeat([[None, None]], start_frame)
            self.op_fill_pts = np.hstack((pad_op_fills, op_fill_pts))
        self.use_op = use_op


    def get_length(self):
        """
        Gets the total number of frames in the video.

        Returns
        -------
        int
            Total number of frames.
        """
        return self.vid_obj.get(cv2.CAP_PROP_FRAME_COUNT)

    def read_frame(self):
        """
        Reads the next frame from the video.

        Returns
        -------
        tuple
            A tuple containing a boolean indicating success and the frame data.
        """
        ret, frame_i = self.vid_obj.read()
        return ret, frame_i

    def set_frame(self, frame_num):
        """
        Sets the current frame number.

        Parameters
        ----------
        frame_num : int
            Frame number to set.
        """
        self.vid_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.current_frame = frame_num

    def set_time_win(self, check_frames):
        """
        Sets the time window for checking urine events.

        Parameters
        ----------
        check_frames : int
            Number of frames to check for urine events.
        """
        if check_frames != self.time_thresh:
            self.time_thresh = check_frames
            self.buffer = PBuffer(self.time_thresh)

    def run_ptect(self, pipe: PtectPipe=None, start_frame=0, end_frame=0, save_path=None, verbose=False):
        """
        Runs the Peetector on the video.

        Parameters
        ----------
        pipe : PtectPipe, optional
            Pipe for sending progress updates (default is None).
        start_frame : int, optional
            Starting frame number (default is 0).
        end_frame : int, optional
            Ending frame number (default is 0).
        save_path : str, optional
            Path to save the output data (default is None).
        verbose : bool, optional
            Whether to print verbose output (default is False).

        Returns
        -------
        tuple or None
            Hot and cool data if save_path is None, otherwise None.
        """
        self.buffer = PBuffer(self.time_thresh)
        self.set_frame(start_frame)
        if end_frame <= 0:
            end_frame = self.total_frames

        all_hot_data = np.empty((0, 3))
        all_cool_data = np.empty((0, 3))

        frame_c = 0
        update_every = 1000
        tot_frames = end_frame - start_frame
        t0 = time.time()
        while self.current_frame < end_frame:
            if verbose and frame_c % update_every == 0 and frame_c > 0:
                t1 = time.time()
                time_diff = (t1 - t0)
                fps = update_every / time_diff
                time_left = (end_frame - self.current_frame) / fps
                time_str = time.strftime('%H:%M:%S', time.gmtime(time_left))
                print(f'Running Ptect on frame {frame_c} of {tot_frames}. Current fps: {fps:.2f} Time left: {time_str}')
                t0 = t1
                # print(frame_c, end_frame, fps, time_left)
                # print(f'Running Ptect on frame {frame_c} of {tot_frames}...')
            if pipe is not None:
                pipe.send((frame_c, tot_frames))
                is_done = pipe.read()
                if is_done:
                    return None
            p_out = self.peetect_next_frame()[0]
            all_hot_data = np.vstack((all_hot_data, p_out[0]))
            all_cool_data = np.vstack((all_cool_data, p_out[1]))
            frame_c += 1

        hot_half = np.hstack((all_hot_data, np.ones_like(all_hot_data[:, 0][:, None])))
        cool_half = np.hstack((all_cool_data, np.zeros_like(all_cool_data[:, 0][:, None])))
        all_data = np.vstack((hot_half, cool_half))

        if save_path is not None:
            np.savez(save_path, urine_data=all_data)
        else:
            return all_hot_data, all_cool_data


    def run_ptect_video(self, start_frame=None, num_frames=None, save_vid=None, show_vid=False, verbose=False):
        """
        Runs the Peetector on the video and optionally saves or displays the output.

        Parameters
        ----------
        start_frame : int, optional
            Starting frame number (default is None).
        num_frames : int, optional
            Number of frames to process (default is None).
        save_vid : str, optional
            Path to save the output video (default is None).
        show_vid : bool, optional
            Whether to display the output video (default is False).
        verbose : bool, optional
            Whether to print verbose output (default is False).

        Returns
        -------
        list
            List of output data for each frame.
        """
        if verbose:
            print('Running Peetect...')

        if start_frame is not None:
            self.set_frame(start_frame)

        out_vid = None
        if save_vid is not None:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out_vid = cv2.VideoWriter(save_vid, fourcc, 120, (1280, 960), isColor=True)

        # if not specified, peetect whole video
        if num_frames is None:
            num_frames = int(self.vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))

        out_acc = []
        for i in range(num_frames):
            if save_vid is not None or show_vid is not None:
                out_data, out_frame = self.peetect_next_frame(return_frame=True)
            else:
                out_data, out_frame = self.peetect_next_frame(return_frame=False)

            if save_vid is not None:
                out_vid.write(out_frame)

            if show_vid:
                cv2.imshow('Peetect Output', out_frame)
                cv2.waitKey(1)
            out_acc.append(out_data)

        if save_vid is not None:
            out_vid.release()
            if verbose:
                print('Peetect video saved')
        cv2.destroyAllWindows()

        if verbose:
            print('Peetect Finished')

        return out_acc


    def peetect_next_frame(self, return_frame=False):
        """
        Processes the next frame for urine events.

        Parameters
        ----------
        return_frame : bool, optional
            Whether to return the processed frame (default is False).

        Returns
        -------
        tuple
            A tuple containing hot data, cool data, and the current frame number.
        """
        hot_thresh = self.heat_thresh
        cool_thresh = self.cool_thresh
        rot_ang = self.rot_ang
        fill_pnts = self.fill_pts
        if self.use_op:
            fill_pnts = []
            for o, t in zip(self.fill_pts, self.op_fill_pts):
                fill_pnts.append(np.vstack((o, t)))

        # collect frame times with urine and urine xys
        urine_evts_times = []
        urine_evts_xys = []
        cool_evts_times = []
        cool_evts_xys = []

        f = self.current_frame

        # run detection on next frame
        is_read, frame_i = self.read_frame()
        out_frame = None
        if is_read:
            this_evts, cool_evts, mask = self.peetect(frame_i, fill_pnts[self.current_frame], cool_thresh=cool_thresh, hot_thresh=hot_thresh)
            self.buffer.push(this_evts)
            good_events = self.buffer.check_ahead()
            # if good urine detected, convert to cm and add to output
            if len(good_events) > 0:
                urine_evts_times.append(f / self.hz)
                urine_evts_xys.append(good_events)

            if len(cool_evts) > 0:
                cool_evts_times.append(f / self.hz)
                cool_evts_xys.append(cool_evts)
                # cool_evts_xys = np.vstack((cool_evts_xys, cool_evts))
                # cool_evts_xys = np.unique(cool_evts_xys, axis=0)


            if return_frame:
                if self.frame_type == 2:
                    output_f = self.show_output(frame_i, good_events, fill_pnts[self.current_frame], cool_evts)
                    out_frame = self.show_all_steps(mask, fill_pnts[self.current_frame], output_f)
                else:
                    out_frame = self.show_output(frame_i, good_events, fill_pnts[self.current_frame], cool_evts)

            self.current_frame += 1

        hot_data = np.empty((0, 3))
        if len(urine_evts_xys) > 0:
            hot_evts_cm = self.urine_px_to_cm_rot(urine_evts_xys, rot_ang=rot_ang)
            hot_data = expand_urine_data(hot_evts_cm, times=urine_evts_times)

        cool_data = np.empty((0, 3))
        if len(cool_evts_xys) > 0:
            cool_evts_cm = self.urine_px_to_cm_rot(cool_evts_xys, rot_ang=rot_ang)
            cool_data = expand_urine_data(cool_evts_cm, times=cool_evts_times)

        out_data = (hot_data, cool_data)
        cur_frame = self.current_frame
        return out_data, out_frame, cur_frame

    def peetect(self, frame, pts, hot_thresh=70, cool_thresh=30):
        """
        Detects urine events in a frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Frame data.
        pts : numpy.ndarray
            Points to fill.
        hot_thresh : int, optional
            Threshold for hot events (default is 70).
        cool_thresh : int, optional
            Threshold for cool events (default is 30).

        Returns
        -------
        tuple
            A tuple containing urine points, cool points, and a list of intermediate masks.
        """

        # get frame dataset, convert to grey
        im_w = np.shape(frame)[1]
        im_h = np.shape(frame)[0]
        f1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # smooth frame
        frame_smooth = self.smooth_frame(f1)

        # mask by thermal threshold
        heat_mask = np.uint8(255 * (frame_smooth > hot_thresh))
        test = heat_mask.copy()

        # dilate resulting mask to hopefully merge mouse parts and expand urine
        di_frame = self.dilate_frame(heat_mask)
        # di_frame = heat_mask

        # fill in all the given points with black
        fill_frame = self.fill_frame_with_points(di_frame, pts, im_w, im_h)

        # erode back previous dilation
        kern = self.dilate_kern
        e_kern = np.ones((kern, kern), np.uint8)
        er_frame = cv2.erode(fill_frame, e_kern, iterations=1)

        # mask valid zone
        valid_frame = self.mask_valid_zone(er_frame)

        # remove deadzones
        dz_frame = self.fill_deadzones(valid_frame)


        # if urine detected set output to urine indices
        urine_xys = []
        if np.sum(dz_frame) > 0:
            urine_xys = np.argwhere(dz_frame > 0)

        # mask valid zone but white
        valid_frame = self.mask_valid_zone(frame_smooth, fill='w')

        # whiten deadzones
        dz_frame_w = self.fill_deadzones(valid_frame, fill='w')

        cool_mask = dz_frame_w < cool_thresh

        # cool xys
        cool_xys = []
        if np.sum(cool_mask) > 0:
            cool_xys = np.argwhere(cool_mask > 0)

        return urine_xys, cool_xys, [frame_smooth, test, di_frame, fill_frame, dz_frame, cool_mask]


    def set_valid_arena(self, shape, *args):
        """
        Sets the valid arena mask.

        Parameters
        ----------
        shape : str
            Shape of the arena ('circle', 'rectangle', or 'polygon').
        *args : tuple
            Additional arguments for the shape (e.g., radius for circle, width and height for rectangle, points for polygon).
        """
        self.valid_zone = make_shape_mask(self.width, self.height, shape,
                                          self.arena_cnt[0], self.arena_cnt[1],*args)
        if shape == 'circle':
            self.arena_params = args[0]

        if shape == 'rectangle':
            self.arena_params = args[:2]

        else:
            self.arena_params = args

        self.arena_shape = shape

    def urine_px_to_cm_rot(self, pts_list, rot_ang=0):
        """
        Converts urine points from pixels to centimeters with rotation.

        Parameters
        ----------
        pts_list : list of numpy.ndarray
            List of urine points in pixels.
        rot_ang : float, optional
            Rotation angle in degrees (default is 0).

        Returns
        -------
        list of numpy.ndarray
            List of urine points in centimeters.
        """
        pts_cm = []
        for pts in pts_list:
            x = (pts[:, 1] - self.arena_cnt[0]) / self.px_per_cm
            y = -(pts[:, 0] - self.arena_cnt[1]) / self.px_per_cm
            # xy = np.vstack((x, y)).T
            x, y = rotate_xy(x, y, rot_ang)
            pts_cm.append(np.vstack((x, y)).T)
        return pts_cm

    def urine_px_to_cm_2d(self, pts):
        """
        Converts urine points from pixels to centimeters.

        Parameters
        ----------
        pts : numpy.ndarray
            Urine points in pixels.

        Returns
        -------
        numpy.ndarray
            Urine points in centimeters.
        """
        x = (pts[:, 0] - self.arena_cnt[0]) / self.px_per_cm
        y = -(pts[:, 1] - self.arena_cnt[1]) / self.px_per_cm
        pts_cm = np.vstack((x, y)).T
        return pts_cm

    def fill_deadzones(self, frame, fill=None):
        """
        Fills dead zones in a frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Frame data.
        fill : str, optional
            Fill color ('w' for white, default is None).

        Returns
        -------
        numpy.ndarray
            Frame with dead zones filled.
        """
        c = (0, 0, 0)
        if fill == 'w':
            c = (255, 255, 255)
        for dz in self.dead_zones:
            dz2 = np.array(dz).astype(int)
            cv2.fillPoly(frame, pts=[dz2], color=c)
        return frame

    def smooth_frame(self, frame):
        """
        Smooths a frame using a kernel.

        Parameters
        ----------
        frame : numpy.ndarray
            Frame data.

        Returns
        -------
        numpy.ndarray
            Smoothed frame.
        """
        s_kern = self.smooth_kern
        smooth_kern = np.ones((s_kern, s_kern), np.float32) / (s_kern * s_kern)
        frame_smooth = cv2.filter2D(src=frame, ddepth=-1, kernel=smooth_kern)
        return frame_smooth

    def dilate_frame(self, frame):
        """
        Dilates a frame using a kernel.

        Parameters
        ----------
        frame : numpy.ndarray
            Frame data.

        Returns
        -------
        numpy.ndarray
            Dilated frame.
        """
        di_kern = self.dilate_kern
        dilate_kern = np.ones((di_kern, di_kern), np.uint8)
        di_frame = cv2.dilate(frame, dilate_kern, iterations=1)
        return di_frame

    def mask_valid_zone(self, frame, fill=None):
        """
        Masks the valid zone in a frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Frame data.
        fill : str, optional
            Fill color ('w' for white, default is None).

        Returns
        -------
        numpy.ndarray
            Frame with valid zone masked.
        """
        valid_frame = cv2.bitwise_and(frame, frame, mask=self.valid_zone)
        if fill == 'w':
            cv2.floodFill(valid_frame, None, (1, 1), 255)
        return valid_frame

    def fill_frame_with_points(self, frame, pnts, width, height):
        """
        Fills a frame with points.

        Parameters
        ----------
        frame : numpy.ndarray
            Frame data.
        pnts : numpy.ndarray
            Points to fill.
        width : int
            Width of the frame.
        height : int
            Height of the frame.

        Returns
        -------
        numpy.ndarray
            Frame with points filled.
        """
        cop_f = frame.copy()
        for p in pnts:
            if p[0] is not None:
                px = int(p[0])
                py = int(p[1])
                if 0 < px < width and 0 < py < height:
                    if frame[py, px] > 0:
                        cv2.floodFill(cop_f, None, (px, py), 0)
        return cop_f

    def add_dz(self, zone=None, num_pts=0):
        """
        Adds dead zones to the arena.

        Parameters
        ----------
        zone : str or list, optional
            Zone type or list of points (default is None).
        num_pts : int, optional
            Number of points to define the dead zone (default is 0).

        Returns
        -------
        list
            List of dead zones.
        """
        w1 = np.array([[316, 210], [330, 210], [330, 480], [316, 480]])
        w2 = np.array([[280, 215], [110, 118], [129, 100], [306, 197]])
        w3 = np.array([[350, 215], [545, 95], [530, 70], [337, 195]]) + [5, 5]
        c_post = np.array([[337, 165], [356, 178], [368, 198], [367, 223], [356, 242], [336, 253], [311, 250],
                           [292, 238], [282, 219], [282, 193], [292, 175], [314, 166]]) + [-2, 3]
        self.dead_zones = []
        if zone == 'block0':
            [self.dead_zones.append(w) for w in [w1, w2, w3, c_post]]
        elif zone == 'block1':
            self.dead_zones.append(c_post)
        else:
            if zone is None:
                vid_obj = cv2.VideoCapture(self.thermal_vid)
                _, frame = vid_obj.read()
                f = plt.figure(label='Define Deadzone')
                plt.imshow(frame)
                if num_pts == 0:
                    num_pts = 10
                plt.title(f'Click {num_pts} times to define deadzone')
                pnts = plt.ginput(n=num_pts, timeout=0)
                plt.close(f)
                this_dz = []
                for p in pnts:
                    this_dz.append([int(p[0]), int(p[1])])
                self.dead_zones.append(this_dz)
            else:
                self.dead_zones = zone

        return self.dead_zones

    def show_all_steps(self, mask_list, pts, out_frame):
        """
        Shows all processing steps for a frame.

        Parameters
        ----------
        mask_list : list of numpy.ndarray
            List of intermediate masks.
        pnts : numpy.ndarray
            Points to fill.

        Returns
        -------
        numpy.ndarray
            Concatenated image of all processing steps.
        """

        # mask_list order: [frame_smooth, heat, di_frame, fill_frame, dz_frame, cool_mask]
        mask_list = [cv2.cvtColor(m.astype(np.uint8), cv2.COLOR_GRAY2BGR) for m in mask_list]

        concat_masks = np.zeros((640, 480))
        mask_h, mask_w, mask_d = np.shape(mask_list[0])
        out_frame = cv2.resize(out_frame, (mask_w, mask_h))
        draw_sleap_pts(mask_list[3], pts)
        draw_zones(mask_list[4], self.arena_shape, self.arena_cnt[0], self.arena_cnt[1], self.arena_params, self.dead_zones)

        if len(mask_list) > 0:
            top_half = np.hstack(mask_list[:3])
            left_half = np.hstack(mask_list[3:-1])
            bot_half = np.hstack((left_half, out_frame))
            concat_masks = np.vstack((top_half, bot_half))
        text_xs = np.array([0.05, 1.05, 2.05, 0.05, 1.05]) * mask_w
        text_ys = np.array([0.95, 0.95, 0.95, 1.95, 1.95]) * mask_h
        text_labs = ['Smooth', 'Heat Thresh', 'Dilate', 'Filled', 'Mask Zones']

        font = ImageFont.truetype(FIRA_MONO, 32)
        img_pil = Image.fromarray(concat_masks)
        draw = ImageDraw.Draw(img_pil)
        for x, y, l in zip(text_xs, text_ys, text_labs):
            draw.rectangle(((int(x) - 8, int(y) - 30), (int(x) + len(l) * 20, int(y) + 5)),
                          (255, 255, 255), -1)
            draw.text((int(x), int(y)), l, font=font, fill=(0, 0, 0, 0), anchor='lb')
        img = np.array(img_pil)
        return img

    def show_output(self, raw_frame, urine_pnts, sleap_pnts, cool_pnts):
        """
        Shows the output frame with annotations.

        Parameters
        ----------
        raw_frame : numpy.ndarray
            Raw frame data.
        urine_pnts : numpy.ndarray
            Urine points.
        sleap_pnts : numpy.ndarray
            SLEAP points.
        cool_pnts : numpy.ndarray
            Cool points.

        Returns
        -------
        numpy.ndarray
            Annotated output frame.
        """
        draw_zones(raw_frame, self.arena_shape, self.arena_cnt[0], self.arena_cnt[1], self.arena_params, self.dead_zones)

        cols = ((0, 1), (1, 2))
        for pnts, c in zip((cool_pnts, urine_pnts), cols):
            if len(pnts) > 0:
                raw_frame[pnts[:, 0], pnts[:, 1], c[0]] = 255
                raw_frame[pnts[:, 0], pnts[:, 1], c[1]] = 255

        draw_sleap_pts(raw_frame, sleap_pnts)

        big_frame = cv2.resize(raw_frame, (1280, 960))
        font = ImageFont.truetype(FIRA_MONO, 48)
        img_pil = Image.fromarray(big_frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((20, 10), 'Fill Points', font=font, fill=(100, 100, 200, 0))
        draw.text((20, 60), 'Dead Zones', font=font, fill=(0, 0, 250, 0))
        draw.text((20, 110), 'Cool Mark', font=font, fill=(255, 255, 0, 0))
        draw.text((20, 160), 'Hot Mark', font=font, fill=(0, 255, 255, 0))
        img = np.array(img_pil)
        return img

def draw_zones(raw_frame, shape, cent_x, cent_y, arena_data, dead_zones):
    match shape:
        case 'circle':
            cv2.circle(raw_frame, (cent_x, cent_y), arena_data[0], (255, 255, 255, 255), 1,
                       cv2.LINE_AA)
        case 'rectangle':
            start_pt = (cent_x - arena_data[0] // 2, cent_y - arena_data[1] // 2)
            end_pt = (cent_x + arena_data[0] // 2, cent_y + arena_data[1] // 2)
            cv2.rectangle(raw_frame, start_pt, end_pt, (255, 255, 255, 255), 1, cv2.LINE_AA)
        case 'custom':
            pts = np.array(arena_data)
            cv2.polylines(raw_frame, [pts], True, (255, 255, 255, 255), 1, cv2.LINE_AA)

    for d in dead_zones:
        pts = np.array(d, dtype=np.int32)
        cv2.polylines(raw_frame, [pts], True, (0, 0, 250), 1, cv2.LINE_AA)

def draw_sleap_pts(raw_frame, sleap_pnts):
    for s in sleap_pnts:
        if s[0] is not None:
            slp_pnt = s.astype(int)
            cv2.circle(raw_frame, (slp_pnt[0], slp_pnt[1]), 3, (100, 100, 200), -1, cv2.LINE_AA)


class PBuffer:
    def __init__(self, buffer_size):
        """
        Initializes the PBuffer.

        Parameters
        ----------
        buffer_size : int
            Size of the buffer.
        """
        buffer_size = max(buffer_size, 1)
        self.size = buffer_size
        self.buffer = np.empty(buffer_size, dtype=np.ndarray)
        self.pos = 0

    def push(self, data):
        """
        Pushes data into the buffer.

        Parameters
        ----------
        data : numpy.ndarray
            Data to push into the buffer.
        """
        self.buffer[self.pos] = data
        self.pos += 1
        if self.pos >= self.size:
            self.pos = 0

    def pop(self):
        """
        Pops data from the buffer.

        Returns
        -------
        numpy.ndarray
            Data from the buffer.
        """
        pos = self.pos - 1
        if pos < 0:
            pos = self.size - 1
        return self.buffer[pos]

    def no_empty(self):
        """
        Checks if the buffer has no empty slots.

        Returns
        -------
        bool
            True if the buffer has no empty slots, False otherwise.
        """
        any_none = np.vectorize(type)(self.buffer)
        any_none = np.any(any_none == NoneType)
        if any_none:
            return False
        lens = np.vectorize(np.size)(self.buffer)
        no_empty = np.all(lens)
        return no_empty

    def check_ahead(self):
        """
        Checks the entire buffer for data present in all slots.

        Returns
        -------
        numpy.ndarray
            Valid events from the buffer.
        """
        if self.size == 1:
            return self.buffer[0]

        true_events = []
        if self.no_empty():
            this_evts = self.pop()
            int_evts = this_evts
            for b in self.buffer:
                int_evts = intersect2d(int_evts, b)
            true_events = int_evts
        return true_events


def urine_across_time(expand_urine, len_s=0, hz=40):
    """
    Computes urine events across time.

    Parameters
    ----------
    expand_urine : numpy.ndarray
        Expanded urine data.
    len_s : float, optional
        Length of the time window in seconds (default is 0).
    hz : int, optional
        Frame rate in Hz (default is 40).

    Returns
    -------
    numpy.ndarray
        Urine events across time.
    """
    urine_over_time = np.empty((0, 2))
    times = expand_urine[:, 0]
    if len_s == 0 and len(times) > 0:
        len_s = np.max(times)
    if len_s > 0:
        unique_ts, urine_cnts = np.unique(times, return_counts=True)
        urine_over_time = np.empty((len(unique_ts), 2))
        urine_over_time[:, 0] = unique_ts.astype(int)
        urine_over_time[:, 1] = urine_cnts
    return urine_over_time


def proj_urine_across_time(urine_data, thresh=0):
    """
    Projects urine events across time.

    Parameters
    ----------
    urine_data : numpy.ndarray
        Urine data.
    thresh : int, optional
        Threshold for urine events (default is 0).

    Returns
    -------
    tuple
        Unique urine points and corresponding times.
    """
    all_xys = urine_data[:, 1:]
    unique_xys, unique_indices, cnts = np.unique(all_xys, axis=0, return_index=True, return_counts=True)
    unique_xys = unique_xys[cnts > thresh, :]
    return unique_xys, all_xys[unique_indices, 0]


def split_urine_data(urine_data):
    """
    Splits urine data into hot and cool events.

    Parameters
    ----------
    urine_data : numpy.ndarray
        Urine data.

    Returns
    -------
    tuple
        Hot and cool urine data.
    """
    hot_ind = urine_data[:, 3].astype(bool)
    return urine_data[hot_ind, :3], urine_data[~hot_ind, :3]


def make_mark_raster(urine_data, hot_thresh=0, cool_thresh=0):
    """
    Creates a raster of urine marks.

    Parameters
    ----------
    urine_data : numpy.ndarray
        Urine data.
    hot_thresh : int, optional
        Threshold for hot events (default is 0).
    cool_thresh : int, optional
        Threshold for cool events (default is 0).

    Returns
    -------
    tuple
        Hot and cool raster data.
    """
    hot_data, cool_data = split_urine_data(urine_data)
    hot_rast = urine_across_time(hot_data, hot_thresh)[:, 0]
    cool_rast = urine_across_time(cool_data, cool_thresh)[:, 1]
    return hot_rast, cool_rast


def urine_segmentation(urine_data, space_dist=1, time_dist=5):
    """
    Segments urine data based on spatial and temporal distance.

    Parameters
    ----------
    urine_data : numpy.ndarray
        Urine data.
    space_dist : float, optional
        Spatial distance threshold (default is 1).
    time_dist : float, optional
        Temporal distance threshold (default is 5).

    Returns
    -------
    numpy.ndarray
        Segmented urine data.
    """
    print('Segmenting urine...')
    time_diff = np.diff(urine_data[:, 0])
    time_clus_ind = np.where(time_diff > time_dist)[0]
    time_clus = np.zeros(len(urine_data))
    for ind, (t0, t1) in enumerate(zip(np.hstack((0, time_clus_ind[:-1])), time_clus_ind)):
        time_clus[t0:t1] = ind

    t_clus = np.unique(time_clus)
    clus_id = np.zeros_like(time_clus)
    clus = 0
    for t_ind, t in enumerate(t_clus):
        t_inds = np.where(time_clus == t)[0]
        t_data = urine_data[t_inds, 1:]
        xys = np.unique(t_data, axis=0)
        t_c = DBSCAN(eps=space_dist, min_samples=1).fit_predict(xys)
        full_clus = np.zeros(len(t_data))
        for i, xy in enumerate(xys):
            inds = np.all(t_data == xy, axis=1)
            full_clus[inds] = t_c[i]
        clus_id[t_inds] = full_clus+clus
        clus += max(full_clus)

    return clus_id


def get_urine_source(mice_cents, urine_data, urine_seg, look_back_frames=40):
    """
    Identifies the source of urine marks.

    Parameters
    ----------
    mice_cents : numpy.ndarray
        Centroid positions of mice.
    urine_data : numpy.ndarray
        Urine data.
    urine_seg : numpy.ndarray
        Segmented urine data.
    look_back_frames : int, optional
        Number of frames to look back (default is 40).

    Returns
    -------
    numpy.ndarray
        Identified urine sources.
    """
    print('Finding Marking Source...')
    num_m = mice_cents.shape[2]
    urine_segs = np.unique(urine_seg)
    urine_ids = np.zeros_like(urine_seg)
    for u in urine_segs:
        if u != np.nan:
            inds = urine_seg == u
            times = urine_data[inds, 0].astype(int)
            cent_seg = np.mean(urine_data[inds, 1:], axis=0)
            mice_xys = mice_cents[times[0]-look_back_frames, :, :]
            dists = np.linalg.norm(mice_xys - cent_seg, axis=1)
            urine_ids[inds] = np.argmin(dists)
    return urine_ids


def dist_to_urine(x, y, expand_urine, thresh=0):
    """
    Computes the distance to the closest urine mark across time.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates.
    y : numpy.ndarray
        Y coordinates.
    expand_urine : numpy.ndarray
        Expanded urine data.
    thresh : int, optional
        Threshold for urine events (default is 0).

    Returns
    -------
    numpy.ndarray
        Distances to urine marks.
    """
    xy_data = np.vstack((x, y)).T
    urine_xy_cm = proj_urine_across_time(expand_urine, thresh=thresh)
    dist_acc = []
    for xy in xy_data:
        xy_vec = np.ones((len(urine_xy_cm), 2)) * np.expand_dims(xy, 1).T
        dists = np.sqrt(np.sum((xy_vec - urine_xy_cm)**2, axis=1))
        dist_acc.append(np.min(dists))
    return np.array(dist_acc)
