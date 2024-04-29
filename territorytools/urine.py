import cv2
import h5py
import matplotlib.patches
from matplotlib.gridspec import GridSpec
from territorytools import behavior as tdt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags


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
    def __init__(self, avi_file, flood_pnts, h_thresh=100, s_kern=5, di_kern=51, t_thresh=20, dead_zones=[], rot_ang=0):
        self.thermal_vid = cv2.VideoCapture(avi_file)
        self.fill_pts = flood_pnts
        self.h_thresh = h_thresh
        self.s_kern = s_kern
        self.di_kern = di_kern
        self.t_thresh = t_thresh
        self.set_dz(dead_zones)

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

    def peetect_frames(self, start_frame=0, num_frames=None, frame_win=20):
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
                urine_evts_xys.append(true_evts)
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
        return urine_evts_times, urine_evts_xys


def get_urine_data(time_file, evt_file):
    times = np.load(time_file)
    evt_xys = np.load(evt_file, allow_pickle=True)
    return times, evt_xys


def urine_px_to_cm(pts, cent_xy=(325, 210), px_per_cm=7.38188976378):
    x = (pts[:, 0] - cent_xy[0]) / px_per_cm
    y = -(pts[:, 1] - cent_xy[1]) / px_per_cm
    return np.vstack((x, y)).T


def plot_urine_xys(xys, ax=None):
    circ = matplotlib.patches.Circle((0, 0), radius=30.48, color=[0.8, 0.8, 0.8])
    ax.add_patch(circ)
    ax.scatter(xys[:, 0], xys[:, 1], marker='+')
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)


def get_mask(evt_xys, thresh=80):
    mask = np.zeros((480, 640))
    for e in evt_xys:
        for pt in e[0]:
            x_ind = pt[1]
            y_ind = pt[0]
            mask[x_ind, y_ind] += 1
    out_m = mask > thresh
    out_m = out_m.astype(int)
    mask_l = np.copy(out_m)
    mask_r = np.copy(out_m)
    mask_h, mask_w = np.shape(out_m)
    mask_l[:, int(mask_w / 2):] = 0
    mask_r[:, :int(mask_w / 2)] = 0
    return out_m, mask_l, mask_r


def get_total_sides(run_data):
    _, mask_l, mask_r = get_mask(run_data)
    left_tot = np.sum(mask_l)
    right_tot = np.sum(mask_r)
    return left_tot, right_tot


def urine_area_over_time(run_data, block=0):
    total_marks_left = np.zeros(90000)
    total_marks_right = np.zeros(90000)
    for t, e in zip(run_data[0], run_data[1]):
        e = e[0]
        total_marks_left[t] = np.sum(e[:, 0] <= 310)
        total_marks_right[t] = np.sum(e[:, 0] > 310)
    total_marks_left = total_marks_left[:72000]
    total_marks_right = total_marks_right[:72000]
    if block == 0:
        return total_marks_left, total_marks_right
    else:
        return total_marks_left + total_marks_right


def dist_to_urine(xy_data, urine_xy_cm):
    # urine_inds = np.argwhere(urine_mask)
    # urine_xy_cm = urine_px_to_cm(np.fliplr(urine_inds))
    dist_acc = []
    for xy in xy_data:
        xy_vec = np.ones((len(urine_xy_cm), 2)) * np.expand_dims(xy, 1).T
        dists = np.sqrt(np.sum((xy_vec - urine_xy_cm)**2, axis=1))
        dist_acc.append(min(dists))
    return np.array(dist_acc)


def marks_per_loc(urine_xys):
    prefs = tdt.compute_preferences(urine_xys)
    return prefs


def explode_urine_data(urine_data):
    times, xys = urine_data[:]
    exploded_data = []
    for t, e in zip(times, xys):
        for xy in e:
            exploded_data.append([t, xy[0], xy[1]])
    return np.array(exploded_data)


def run_cc(g_id, g_data, g_info):
    for g, g_i in zip(g_data, g_info):
        u_r, u_i = urine_area_over_time(g)
        corr = correlate(u_r, u_i)
        lags = correlation_lags(len(u_r), len(u_i))
        corr /= np.max(corr)
        plt.figure()
        plt.plot(lags / (40 * 60), corr)
        f_name = g_i['Resident'] + ' vs ' + g_i['Intruder'] + ' Day ' + g_i['Day']
        plt.title(f_name)
        plt.show()


def plot_all_series_and_mean(g_id, g_data, g_info):
    f, axs = plt.subplots(2, 1, figsize=(20, 10))
    m0 = g_data[0]
    num_f = len(m0[0])
    temp = np.zeros((num_f, len(g_data) * 2))
    c = -1
    for m in g_data:
        c += 1
        t = np.arange(num_f) / (40 * 60)
        axs[0].plot(t, m[0], c=[0.5, 0.5, 0.5])
        axs[0].plot(t, m[1], c=[0.5, 0.5, 0.5])
        temp[:, c] = m[0]
        temp[:, c + 1] = m[1]
    axs[1].plot(t, np.mean(temp, axis=1), c='r')
    plt.show()
    print('yes')


def plot_all_masks(g_id, g_data, g_info):
    f, axs = plt.subplots(1, 1, figsize=(10, 10))
    m0 = g_data[0]
    mask = np.zeros_like(m0)
    for g in g_data:
        mask += g
    axs.imshow(mask)
    axs.set_title(g_id)


def plot_all_urine_xys(g_id, g_data, g_info, params):
    c, ax = params[:]
    group_n = len(g_data)
    for g in g_data:
        ax.scatter(g[:, 0], g[:, 1], color=c, s=1)
    ax.set_title('Group: ' + g_id)
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    return None


def urine_raster_by_territory(urine_data):
    exploded = explode_urine_data(urine_data)
    _, r_marks, i_marks, n_marks = tdt.compute_preferences((exploded[:, 1], exploded[:, 2]))
    r_times = np.unique(exploded[r_marks, 0])
    i_times = np.unique(exploded[i_marks, 0])
    n_times = np.unique(exploded[n_marks, 0])
    fig, axs = plt.subplots(3, 1)
    for i, t in enumerate((r_times, i_times, n_times)):
        axs[i].stem(t, np.ones(len(t)))
        axs[i].set_xlim(0, 72000)
    plt.show()


def plot_block0(run_data):
    times, evt_xys = run_data[:]
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    gs = GridSpec(2, 4, figure=fig)
    total_marks_left, total_marks_right = urine_area_over_time(run_data)
    times = np.arange(len(total_marks_left)) / (40 * 60)
    y1 = max(np.max(total_marks_left), np.max(total_marks_right))
    y1 = y1 + 0.2 * y1
    y1 = 1200
    ax0 = fig.add_subplot(gs[1, 2:])
    ax0.plot(times, total_marks_right, c=[1, 0.5, 0])
    ax0.set_ylim(0, y1)
    ax0.set_xlabel('Time (min)')
    ax0.set_ylabel('Urine Area (px)')
    ax1 = fig.add_subplot(gs[0, 2:])
    ax1.plot(times, total_marks_left, c=[0, 0.8, 0])
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Urine Area (px)')
    ax1.set_ylim(0, y1)
    _, mask_l, mask_r = get_mask(run_data)
    mask_h, mask_w = np.shape(mask_l)
    rgb = np.zeros((mask_h, mask_w, 3))
    rgb[:, :, 1] += mask_l
    rgb[:, :, 0] += mask_r
    rgb[:, :, 1] += mask_r / 2
    ax2 = fig.add_subplot(gs[:, :2])
    ax2.imshow(np.flipud(rgb))
    plt.show()
