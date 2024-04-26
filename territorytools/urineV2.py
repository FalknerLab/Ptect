import cv2
import h5py
import matplotlib.patches
from territorytools.behavior import compute_preferences, rotate_xy
import numpy as np
import matplotlib.pyplot as plt


def sleap_to_fill_pts(sleap_h5):
    with h5py.File(sleap_h5, "r") as f:
        locations = f["tracks"][:].T
    t, d1, d2, d3 = np.shape(locations)
    fill_pts = []
    for i in range(t):
        t_pts = locations[i, :, :, :]
        t_pts = np.moveaxis(t_pts, 0, 1)
        t_pts = np.reshape(t_pts, (d2, d1 * d3))
        keep = ~np.all(np.isnan(t_pts), axis=0)
        k_pts = t_pts[:, keep]
        fill_pts.append(k_pts.T)
    return fill_pts


def check_px_across_window(this_evts, win_evts):
    true_events = []
    bool_acc = np.ones(np.shape(this_evts)[0])
    for e in win_evts:
        if len(e) == 0:
            return []
        str_xys = np.char.add(this_evts.astype(str)[:, 0], this_evts.astype(str)[:, 1])
        str_exys = np.char.add(e.astype(str)[:, 0], e.astype(str)[:, 1])
        all_next = np.isin(str_xys, str_exys)
        bool_acc = np.logical_and(bool_acc, all_next)
    keep_inds = bool_acc
    if np.any(keep_inds):
        good_evts = this_evts[keep_inds]
        good_evts = np.fliplr(good_evts)
        true_events = good_evts
    return true_events


def urine_px_to_cm(pts, cent_xy=(325, 210), px_per_cm=7.38188976378):
    x = (pts[:, 0] - cent_xy[0]) / px_per_cm
    y = -(pts[:, 1] - cent_xy[1]) / px_per_cm
    return np.vstack((x, y)).T


class Peetector:
    def __init__(self, avi_file, flood_pnts, h_thresh=100, s_kern=5, di_kern=51, t_thresh=20, dead_zones=[], cent_xy=(325, 210), px_per_cm=7.38188976378):
        self.thermal_vid = avi_file
        if type(flood_pnts) == 'str':
            self.fill_pts = sleap_to_fill_pts(flood_pnts)
        else:
            self.fill_pts = flood_pnts
        self.h_thresh = h_thresh
        self.s_kern = s_kern
        self.di_kern = di_kern
        self.t_thresh = t_thresh
        self.dead_zones = dead_zones
        self.arena_cnt = cent_xy
        self.px_per_cm = px_per_cm


    def remove_dz(self, mask):
        for dz in self.dead_zones:
            cv2.fillPoly(mask, pts=[dz], color=(255, 255, 255))


    def peetect(self, frame, pts):
        # get frame data, convert to grey
        im_w = np.shape(frame)[1]
        im_h = np.shape(frame)[0]
        f1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # smooth frame
        smooth_kern = np.ones((self.s_kern, self.s_kern), np.float32) / (self.s_kern * self.s_kern)
        frame_smooth = cv2.filter2D(src=f1, ddepth=-1, kernel=smooth_kern)

        # mask by thermal threshold
        mask = np.uint8(255 * (frame_smooth > self.h_thresh))

        # remove deadzones
        remove_dz(mask)

        # dilate resulting mask to hopefully merge mouse parts and expand urine
        dilate_kern = np.ones((self.di_kern, self.di_kern), np.uint8)
        cv2.dilate(mask, dilate_kern, iterations=1)

        # fill in all the given points with black
        urine_xys = []
        for p in pts:
            px = int(p[0])
            py = int(p[1])
            if 0 < px < im_w and 0 < py < im_h:
                if mask[py, px] > 0:
                    cv2.floodFill(mask, None, (int(p[0]), int(p[1])), 0)

        # if urine detected set output to urine indices
        if np.sum(mask) > 0:
            urine_xys = np.argwhere(mask > 0)
        return urine_xys

    def peetect_frames(self, start_frame=0, num_frames=None, frame_win=20, save_path=None):
        # if not specified, peetect whole video
        if num_frames is None:
            num_frames = int(self.thermal_vid.get(cv2.CAP_PROP_FRAME_COUNT))

        # setup video object at starting frame
        vid_obj = cv2.VideoCapture(self.thermal_vid)
        vid_obj.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # initialize first window of urine events to check if detected urine stays hot long enough
        win_buf = []
        for i in range(frame_win):
            is_read, frame_i = vid_obj.read()
            evts = []
            if is_read:
                evts = self.peetect(frame_i, self.fill_pts[i])
            win_buf.append(evts)

        # collect frame times with urine and urine xys
        urine_evts_times = []
        urine_evts_xys = []

        for f in range(frame_win, num_frames):
            # check first frame in buffer to see if any urine points stay hot
            true_evts = check_px_across_window(win_buf[0], win_buf)

            # if good urine detected, convert to cm and add to output
            if len(true_evts) > 0:
                urine_evts_times.append(f - frame_win)
                urine_xys = urine_px_to_cm(true_evts, cent_xy=self.arena_cnt, px_per_cm=self.px_per_cm)
                urine_xys = rotate_xy(urine_xys[:, 0], urine_xys[:, 1], self.rot)
                urine_evts_xys.append(urine_xys)

            # run detection on next frame
            is_read, frame_i = vid_obj.read()
            this_evts = []
            if is_read:
                evts = self.peetect(frame_i, self.fill_pts[f])
                if len(evts) > 0:
                    this_evts = evts

            # add to buffer and remove oldest frame
            win_buf.append(this_evts)
            win_buf.pop(0)

            if f % 1000 == 0:
                print('Peetect running, on frame: ', srt(f))

        return urine_evts_times, urine_evts_xys

    def add_dz(self):
        vid_obj = cv2.VideoCapture(self.thermal_vid)
        _, frame = vid_obj.read()
        plt.figure()
        plt.imshow(frame)
        pnts = plt.ginput(n=10, timeout=600)
        self.dead_zones.append(pnts)
        return pnts


# def set_dz(self, dead_zones):
#     if dead_zones == "Block0":
#         w1 = np.array([[316, 210], [330, 210], [330, 480], [316, 480]])
#         w2 = np.array([[311, 212], [111, 111], [129, 85], [306, 197]])
#         w3 = np.array([[340, 205], [577, 104], [540, 80], [337, 195]])
#         self.dead_zones = (w1, w2, w3)
#     else:
#         self.dead_zones = dead_zones
