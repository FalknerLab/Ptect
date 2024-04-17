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
        true_events.append(good_evts)
    return true_events


class Peetector:
    def __init__(self, avi_file, flood_pnts, h_thresh=100, s_kern=5, di_kern=51, t_thresh=20, dead_zones=[], rot_ang=0, cent_xy=(325, 210), px_per_cm=7.38188976378):
        self.thermal_vid = cv2.VideoCapture(avi_file)
        if type(flood_pnts) == 'str':
            self.fill_pts = sleap_to_fill_pts(flood_pnts)
        else:
            self.fill_pts = flood_pnts
        self.h_thresh = h_thresh
        self.s_kern = s_kern
        self.di_kern = di_kern
        self.t_thresh = t_thresh
        self.set_dz(dead_zones)
        self.arena_cnt = cent_xy
        self.rot = rot_ang
        self.px_per_cm = px_per_cm

    def set_dz(self, dead_zones):
        if dead_zones == "Block0":
            w1 = np.array([[316, 210], [330, 210], [330, 480], [316, 480]])
            w2 = np.array([[311, 212], [111, 111], [129, 85], [306, 197]])
            w3 = np.array([[340, 205], [577, 104], [540, 80], [337, 195]])
            self.dead_zones = (w1, w2, w3)
        else:
            self.dead_zones = dead_zones

    def peetect(self, frame, pts):
        im_w = np.shape(frame)[1]
        im_h = np.shape(frame)[0]
        kern = np.ones((self.s_kern, self.s_kern), np.float32) / (self.s_kern * self.s_kern)
        kern2 = np.ones((self.di_kern, self.di_kern), np.uint8)
        f1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameg = cv2.filter2D(src=f1, ddepth=-1, kernel=kern)
        cv2.circle(frameg, (320, 208), 50, 255, -1)
        mask_frame = np.zeros_like(frameg)
        cv2.circle(mask_frame, (325, 230), 200, 255, -1)
        frame = cv2.bitwise_and(frameg, frameg, mask=mask_frame)
        mask = np.uint8(255 * (frame > self.h_thresh))
        for dz in self.dead_zones:
            cv2.fillPoly(mask, pts=[dz], color=(255, 255, 255))
        cv2.dilate(mask, kern2, iterations=1)
        cv2.floodFill(mask, None, (320, 208), 0)
        urine_xys = []
        for p in pts:
            px = int(p[0])
            py = int(p[1])
            if 0 < px < im_w and 0 < py < im_h:
                if mask[py, px] > 0:
                    cv2.floodFill(mask, None, (int(p[0]), int(p[1])), 0)
        if np.sum(mask) > 0:
            urine_xys = np.argwhere(mask > 0)
        return urine_xys

    def peetect_frames(self, start_frame=0, num_frames=None, frame_win=20, save_path=None):
        if num_frames is None:
            num_frames = int(self.thermal_vid.get(cv2.CAP_PROP_FRAME_COUNT))

        vid_obj = self.thermal_vid
        vid_obj.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        win_buf = []
        for i in range(frame_win):
            is_read, frame_i = vid_obj.read()
            evts = []
            if is_read:
                evts = self.peetect(frame_i, self.fill_pts[i])
            win_buf.append(evts)

        urine_evts_times = []
        urine_evts_xys = []
        for f in range(frame_win, num_frames):
            true_evts = check_px_across_window(win_buf[0], win_buf)
            if len(true_evts) > 0:
                urine_evts_times.append(f - frame_win)
                print(true_evts)
                urine_xys = urine_px_to_cm(true_evts, cent_xy=self.arena_cnt, px_per_cm=self.px_per_cm)
                urine_xys = rotate_xy(urine_xys[:, 0], urine_xys[:, 1], self.rot)
                urine_evts_xys.append(urine_xys)
            is_read, frame_i = vid_obj.read()
            this_evts = []
            if is_read:
                evts = self.peetect(frame_i, self.fill_pts[f])
                if len(evts) > 0:
                    this_evts = evts
            win_buf.append(this_evts)
            win_buf.pop(0)
            if f % 250 == 0:
                print(f)
        if save_path is not None:
            np.save(save_path + '-urine_times.npy', urine_evts_times)
            np.save(save_path + '-urine_xys.npy', urine_evts_xys)
        return urine_evts_times, urine_evts_xys


def load_urine_data(time_file, evt_file):
    times = np.load(time_file)
    evt_xys = np.load(evt_file, allow_pickle=True)
    return times, evt_xys


def urine_px_to_cm(pts, cent_xy=(325, 210), px_per_cm=7.38188976378):
    pts_np = np.array(pts)
    x = (pts_np[:, 0] - cent_xy[0]) / px_per_cm
    y = -(pts_np[:, 1] - cent_xy[1]) / px_per_cm
    return np.vstack((x, y)).T


def get_unique_marks(times_data, evt_data, thresh=80):
    u_exp = explode_urine_data(times_data, evt_data)[:, 1:]
    u_u, u_cnts = np.unique(u_exp, axis=0, return_counts=True)
    g_uxy = u_u[u_cnts > thresh]
    return g_uxy


def plot_urine_xys(xys, ax=None):

    circ = matplotlib.patches.Circle((0, 0), radius=30.48, color=[0.8, 0.8, 0.8])
    ax.add_patch(circ)
    ax.scatter(xys[:, 0], xys[:, 1], marker='+')
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)


def get_total_sides(times, urine_xys):
    unique_xys = get_unique_marks(times, urine_xys)
    left_tot = np.sum(unique_xys[:, 0] < 0)
    right_tot = np.sum(unique_xys[:, 0] > 0)
    return left_tot, right_tot


def urine_area_over_time(times, urine_xys, block=0, ts_len=0):
    total_marks_left = np.zeros(ts_len)
    total_marks_right = np.zeros(ts_len)
    for t, e in zip(times, urine_xys):
        e = e[0]
        total_marks_left[t] = np.sum(e[:, 0] < 0)
        total_marks_right[t] = np.sum(e[:, 0] > 0)
    total_marks_left = total_marks_left[:ts_len]
    total_marks_right = total_marks_right[:ts_len]
    if block == 0:
        return total_marks_left, total_marks_right
    else:
        return total_marks_left + total_marks_right


def dist_to_urine(xy_data, urine_xy_cm):
    dist_acc = []
    for xy in xy_data:
        xy_vec = np.ones((len(urine_xy_cm), 2)) * np.expand_dims(xy, 1).T
        dists = np.sqrt(np.sum((xy_vec - urine_xy_cm)**2, axis=1))
        dist_acc.append(min(dists))
    return np.array(dist_acc)


def marks_per_loc(urine_xys):
    prefs = compute_preferences(urine_xys)
    return prefs


def explode_urine_data(times, urine_xys):
    exploded_data = []
    for t, e in zip(times, urine_xys):
        for xy in e:
            exploded_data.append([t, xy[0], xy[1]])
    return np.array(exploded_data)


def urine_raster_by_territory(times, urine_xys):
    exploded = explode_urine_data(times, urine_xys)
    _, r_marks, i_marks, n_marks = compute_preferences((exploded[:, 1], exploded[:, 2]))
    r_times = np.unique(exploded[r_marks, 0])
    i_times = np.unique(exploded[i_marks, 0])
    n_times = np.unique(exploded[n_marks, 0])
    fig, axs = plt.subplots(3, 1)
    for i, t in enumerate((r_times, i_times, n_times)):
        axs[i].stem(t, np.ones(len(t)))
