import cv2
import h5py
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
        if type(flood_pnts) == str:
            print('Converting slp file to fill points...')
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
            dz2 = np.array(dz).astype(int)
            cv2.fillPoly(mask, pts=[dz2], color=(0, 0, 0))


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
        self.remove_dz(mask)

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
                    cv2.floodFill(mask, None, (px, py), 0)
                    cv2.circle(mask, (px, py), 10, 255)

        # if urine detected set output to urine indices
        if np.sum(mask) > 0:
            urine_xys = np.argwhere(mask > 0)
        return urine_xys, mask

    def peetect_frames(self, start_frame=0, num_frames=None, frame_win=80, save_path=None):
        # setup video object
        vid_obj = cv2.VideoCapture(self.thermal_vid)

        # if not specified, peetect whole video
        if num_frames is None:
            num_frames = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))

        vid_obj.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # initialize first window of urine events to check if detected urine stays hot long enough
        win_buf = []
        frame_buf = []
        mask_buf = []
        fillpt_buf = []
        for i in range(frame_win):
            is_read, frame_i = vid_obj.read()
            evts = []
            mask = []
            if is_read:
                evts, mask = self.peetect(frame_i, self.fill_pts[i])
            mask_buf.append(mask)
            win_buf.append(evts)
            frame_buf.append(frame_i)
            fillpt_buf.append(self.fill_pts[i])

        # collect frame times with urine and urine xys
        urine_evts_times = []
        urine_evts_xys = []
        print('Running Peetect...')
        for f in range(frame_win, num_frames):
            # check oldest frame in buffer to see if any urine points stay hot
            true_evts = check_px_across_window(win_buf[0], win_buf)

            # if good urine detected, convert to cm and add to output
            if len(true_evts) > 0:
                urine_evts_times.append(f - frame_win)
                urine_xys = urine_px_to_cm(true_evts, cent_xy=self.arena_cnt, px_per_cm=self.px_per_cm)
                urine_evts_xys.append(urine_xys)

            show_output(frame_buf[0], mask_buf[0], true_evts, fillpt_buf[0])

            # run detection on next frame
            is_read, frame_i = vid_obj.read()
            this_evts = []
            mask = []
            if is_read:
                evts, mask = self.peetect(frame_i, self.fill_pts[f])
                if len(evts) > 0:
                    this_evts = evts

            # add to buffer and remove oldest frame
            win_buf.append(this_evts)
            win_buf.pop(0)
            frame_buf.append(frame_i)
            frame_buf.pop(0)
            mask_buf.append(mask)
            mask_buf.pop(0)
            fillpt_buf.append(self.fill_pts[f])
            fillpt_buf.pop(0)
            if f % 1 == 0:
                print('Peetect running, on frame: ', str(f))

        return urine_evts_times, urine_evts_xys

    def add_dz(self, zone=None, num_pts=0):
        pnts = []
        if zone is None:
            vid_obj = cv2.VideoCapture(self.thermal_vid)
            _, frame = vid_obj.read()
            plt.figure()
            plt.imshow(frame)
            if num_pts == 0:
                num_pts = 10
            pnts = plt.ginput(n=num_pts, timeout=600)
        else:
            pnts = zone
        self.dead_zones.append(pnts)
        return pnts


def show_output(raw_frame, mask_frame, urine_pnts, sleap_pnts):
    for i in urine_pnts:
        cv2.circle(raw_frame, i, 1, (0, 200, 100))
    # col_mask = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
    for s in sleap_pnts:
        slp_pnt = s.astype(int)
        cv2.circle(raw_frame, (slp_pnt[0], slp_pnt[1]), 10, (0, 100, 200))
    # c_frame = cv2.hconcat([raw_frame, col_mask])
    cv2.imshow('Frame', mask_frame)
    cv2.waitKey(1)

# def set_dz(self, dead_zones):
#     if dead_zones == "Block0":
#         w1 = np.array([[316, 210], [330, 210], [330, 480], [316, 480]])
#         w2 = np.array([[311, 212], [111, 111], [129, 85], [306, 197]])
#         w3 = np.array([[340, 205], [577, 104], [540, 80], [337, 195]])
#         self.dead_zones = (w1, w2, w3)
#     else:
#         self.dead_zones = dead_zones
