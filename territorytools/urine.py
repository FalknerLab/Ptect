import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib.animation import FuncAnimation
from matplotlib import cm


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
    def __init__(self, avi_file, flood_pnts, h_thresh=70, s_kern=5, di_kern=51, t_thresh=20, dead_zones=[],
                 cent_xy=(320, 212), px_per_cm=7.38188976378):
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

    def fill_deadzones(self, mask):
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
        circ_mask = cv2.circle(np.zeros_like(frame_smooth), (self.arena_cnt[0] + 12, self.arena_cnt[1] + 18), 195, 255,
                               -1)
        frame_smooth = cv2.bitwise_and(frame_smooth, frame_smooth, mask=circ_mask)

        # mask by thermal threshold
        mask = np.uint8(255 * (frame_smooth > self.h_thresh))

        # remove deadzones
        self.fill_deadzones(mask)

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

    def peetect_frames(self, start_frame=0, num_frames=None, frame_win=80, save_path=None, save_vid=None, show_vid=False):
        # setup video object
        vid_obj = cv2.VideoCapture(self.thermal_vid)

        # if not specified, peetect whole video
        if num_frames is None:
            num_frames = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))

        # offset video and fill points to start frame
        vid_obj.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fill_pnts = self.fill_pts[start_frame:]

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
                evts, mask = self.peetect(frame_i, fill_pnts[i])
            mask_buf.append(mask)
            win_buf.append(evts)
            frame_buf.append(frame_i)
            fillpt_buf.append(fill_pnts[i])

        out_vid = None
        if save_vid is not None:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out_vid = cv2.VideoWriter(save_vid, fourcc, 120, (frame_i.shape[1], frame_i.shape[0]), isColor=True)

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

            if show_vid or save_vid is not None:
                out_frame = self.show_output(frame_buf[0], true_evts, fillpt_buf[0])

            if save_vid is not None:
                out_vid.write(out_frame)

            if show_vid:
                cv2.imshow('Peetect Output', out_frame)
                cv2.waitKey(1)

            # run detection on next frame
            is_read, frame_i = vid_obj.read()
            this_evts = []
            mask = []
            if is_read:
                evts, mask = self.peetect(frame_i, fill_pnts[f])
                if len(evts) > 0:
                    this_evts = evts

            # add to buffer and remove oldest frame
            win_buf.append(this_evts)
            win_buf.pop(0)
            frame_buf.append(frame_i)
            frame_buf.pop(0)
            mask_buf.append(mask)
            mask_buf.pop(0)
            fillpt_buf.append(fill_pnts[f])
            fillpt_buf.pop(0)
            if f % 100 == 0:
                print('Peetect running, on frame: ', str(f))
        if save_vid is not None:
            out_vid.release()
            print('Peetect video saved')
            cv2.destroyAllWindows()
        return urine_evts_times, urine_evts_xys

    def add_dz(self, zone=None, num_pts=0):
        w1 = np.array([[316, 210], [330, 210], [330, 480], [316, 480]])
        w2 = np.array([[311, 225], [111, 125], [129, 85], [306, 197]])
        w3 = np.array([[340, 215], [577, 104], [530, 70], [337, 195]])
        c_post = np.array([[337, 165], [356, 178], [368, 198], [367, 223], [356, 242], [336, 253], [311, 250],
                           [292, 238], [282, 219], [282, 193], [292, 175], [314, 166]]) + [-5, 5]
        if zone == 'block0':
            [self.dead_zones.append(w) for w in [w1, w2, w3, c_post]]
        elif zone == 'block1':
            self.dead_zones.append(c_post)
        else:
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

    def show_output(self, raw_frame, urine_pnts, sleap_pnts):
        # mask_c = cv2.circle(np.zeros_like(raw_frame), (self.arena_cnt[0] + 12, self.arena_cnt[1] + 18), 195, 255, -1)
        # raw_frame = cv2.bitwise_and(raw_frame, raw_frame, mask=mask_c[:, :, 0])
        cv2.circle(raw_frame, (self.arena_cnt[0] + 12, self.arena_cnt[1] + 18), 195, 255, 1)
        urine_mask = np.zeros_like(raw_frame[:, :, 0])
        if len(urine_pnts) > 0:
            urine_mask[urine_pnts[:, 1], urine_pnts[:, 0]] = 1
            contours, heir = cv2.findContours(urine_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(raw_frame, contours, -1, (0, 255, 0), 1)
        for s in sleap_pnts:
            slp_pnt = s.astype(int)
            cv2.circle(raw_frame, (slp_pnt[0], slp_pnt[1]), 3, (0, 100, 200))
        for d in self.dead_zones:
            cv2.polylines(raw_frame, [d], True, (0, 0, 250))
        cv2.putText(raw_frame, 'Fill Points', (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 100, 200))
        cv2.putText(raw_frame, 'Dead Zones', (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 250))
        cv2.putText(raw_frame, 'Mark Detected', (10, 75), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 200, 100))
        return raw_frame


def proj_urine_across_time(urine_xys, thresh=0):
    all_xys = expand_urine_data(urine_xys)
    unique_xys, cnts = np.unique(all_xys, axis=0, return_counts=True)
    unique_xys = unique_xys[cnts > thresh, :]
    return unique_xys


def expand_urine_data(urine_xys, times=None):
    def make_ts(t, n):
        return t * np.ones(n)

    num_urine_pnts_per_t = np.vectorize(len)(urine_xys)
    expanded_data = np.vstack(urine_xys)
    if times is not None:
        time_vec = np.zeros(expanded_data.shape[0])
        c = 0
        for ind, (t, nums) in enumerate(zip(times, num_urine_pnts_per_t)):
            time_vec[c:(c + nums)] = t
            c += nums
        expanded_data = np.hstack((time_vec[:, None], expanded_data))
    return expanded_data


def urine_segmentation(times_data, urine_xys, do_animation=False):
    urine_data = expand_urine_data(urine_xys, times=times_data)
    print('Segmenting urine...')
    clustering = DBSCAN(eps=2, min_samples=5).fit(urine_data)
    clus_id = clustering.labels_
    if do_animation:
        show_urine_segmented(urine_data, clus_id)
    # else:
    #     ax = plt.subplot(projection='3d')
    #     ax.scatter(urine_data[:, 0], urine_data[:, 1], urine_data[:, 2], c=clus_id, cmap='Set3')
    #     plt.show()
    return clus_id


def show_urine_segmented(expand_urine, labels, window=300):
    num_f = int(max(expand_urine[:, 0])) + window
    f = plt.figure()
    ax = f.add_subplot(projection='3d')
    ax.set_xlim(0, window)
    ax.set_ylim(-32, 32)
    ax.set_zlim(-32, 32)
    s = ax.scatter([], [], [])
    set3 = cm.get_cmap('Set3', max(labels)).colors

    def update(frame):
        data_in_win = np.logical_and(expand_urine[:, 0] > frame, expand_urine[:, 0] < frame + window)
        frame_data = expand_urine[data_in_win, :]
        s._offsets3d = (frame_data[:, 0] - frame, frame_data[:, 1], frame_data[:, 2])
        s.set_color(set3[labels[data_in_win], :])
        return s

    print('Running animation...')
    anim = FuncAnimation(fig=f, func=update, frames=num_f, interval=1)
    anim.save('urine3d.mp4', writer='ffmpeg', progress_callback=lambda i, n: print(f'Saving frame {i}/{n}'), fps=30)


def get_urine_source(test_cents, exp_urine):
    urine_ids = []
    for i in range(len(exp_urine)):
        this_t = exp_urine[i, 0]
        urine_xy = exp_urine[i, 1:]
        dists = np.linalg.norm(test_cents[this_t, :, :] - urine_xy, axis=0)
        urine_ids.append(np.argmin(dists))
    return urine_ids
