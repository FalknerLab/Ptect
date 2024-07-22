import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from PIL import ImageFont, ImageDraw, Image


def sleap_to_fill_pts(sleap_h5):
    with h5py.File(sleap_h5, "r") as f:
        locations = f["tracks"][:].T
    t, d1, d2, d3 = np.shape(locations)
    fill_pts = []
    last_pts = []
    for i in range(t):
        t_pts = locations[i, :, :, :]
        t_pts = np.moveaxis(t_pts, 0, 1)
        t_pts = np.reshape(t_pts, (d2, d1 * d3))
        keep = ~np.all(np.isnan(t_pts), axis=0)
        if np.any(keep):
            k_pts = t_pts[:, keep]
            fill_pts.append(k_pts.T)
            last_pts = k_pts
        else:
            fill_pts.append(last_pts.T)
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


def expand_urine_data(urine_xys, times=None):
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


class Peetector:
    def __init__(self, avi_file, flood_pnts, dead_zones=[], cent_xy=(320, 212), px_per_cm=7.38188976378,
                 heat_thresh=70, s_kern=5, di_kern=5, v_mask=None):
        self.thermal_vid = avi_file
        vid_obj = cv2.VideoCapture(avi_file)
        width = int(vid_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if type(flood_pnts) == str:
            print('Converting slp file to fill points...')
            self.fill_pts = sleap_to_fill_pts(flood_pnts)
        else:
            self.fill_pts = flood_pnts
        self.dead_zones = dead_zones
        self.arena_cnt = cent_xy
        self.px_per_cm = px_per_cm
        self.heat_thresh = heat_thresh
        self.smooth_kern = s_kern
        self.dilate_kern = di_kern
        if v_mask is None:
            self.valid_zone = cv2.circle(np.zeros((height, width)), (self.arena_cnt[0] + 12, self.arena_cnt[1] + 18),
                                    195, 255, -1)
            self.valid_zone = self.valid_zone.astype('uint8')
        else:
            self.valid_zone = v_mask

    def fill_deadzones(self, frame):
        for dz in self.dead_zones:
            dz2 = np.array(dz).astype(int)
            cv2.fillPoly(frame, pts=[dz2], color=(0, 0, 0))
        return frame

    def smooth_frame(self, frame):
        s_kern = self.smooth_kern
        smooth_kern = np.ones((s_kern, s_kern), np.float32) / (s_kern * s_kern)
        frame_smooth = cv2.filter2D(src=frame, ddepth=-1, kernel=smooth_kern)
        return frame_smooth

    def dilate_frame(self, frame):
        di_kern = self.dilate_kern
        dilate_kern = np.ones((di_kern, di_kern), np.uint8)
        di_frame = cv2.dilate(frame, dilate_kern, iterations=1)
        return di_frame

    def mask_valid_zone(self, frame):
        valid_frame = cv2.bitwise_and(frame, frame, mask=self.valid_zone)
        return valid_frame

    def fill_frame_with_points(self, frame, pnts, width, height):
        cop_f = frame.copy()
        for p in pnts:
            px = int(p[0])
            py = int(p[1])
            if 0 < px < width and 0 < py < height:
                if frame[py, px] > 0:
                    cv2.floodFill(cop_f, None, (px, py), 0)
                    # cv2.circle(frame, (px, py), 10, 255)
        return cop_f

    def peetect(self, frame, pts):
        # get frame data, convert to grey
        im_w = np.shape(frame)[1]
        im_h = np.shape(frame)[0]
        f1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # smooth frame
        frame_smooth = self.smooth_frame(f1)

        # mask valid zone
        valid_frame = self.mask_valid_zone(frame_smooth)

        # remove deadzones
        dz_frame = self.fill_deadzones(valid_frame)

        # mask by thermal threshold
        heat_mask = np.uint8(255 * (dz_frame > self.heat_thresh))
        test = heat_mask.copy()

        # dilate resulting mask to hopefully merge mouse parts and expand urine
        di_frame = self.dilate_frame(heat_mask)

        # fill in all the given points with black
        fill_frame = self.fill_frame_with_points(di_frame, pts, im_w, im_h)

        # if urine detected set output to urine indices
        urine_xys = []
        if np.sum(fill_frame) > 0:
            urine_xys = np.argwhere(fill_frame > 0)

        return urine_xys, [f1, frame_smooth, dz_frame, test, di_frame, fill_frame]

    def peetect_frames(self, start_frame=0, num_frames=None, time_thresh=2,
                       save_vid=None, show_vid=False, hz=40):

        # setup video object
        vid_obj = cv2.VideoCapture(self.thermal_vid)

        # if not specified, peetect whole video
        if num_frames is None:
            num_frames = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))

        # offset video and fill points to start frame
        vid_obj.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fill_pnts = self.fill_pts[start_frame:]

        frame_win = int(time_thresh*hz)

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
            # out_vid = cv2.VideoWriter(save_vid, fourcc, 120, (frame_i.shape[1], frame_i.shape[0]), isColor=True)
            out_vid = cv2.VideoWriter(save_vid, fourcc, 120, (1280, 960), isColor=True)

        # collect frame times with urine and urine xys
        urine_evts_times = []
        urine_evts_xys = []
        print('Running Peetect...')
        for f in range(frame_win, num_frames):
            if f % 100 == 0:
                print('Running Peetect on frame: ', f, ' of ', num_frames)

            # check oldest frame in buffer to see if any urine points stay hot
            true_evts = check_px_across_window(win_buf[0], win_buf)

            # if good urine detected, convert to cm and add to output
            if len(true_evts) > 0:
                urine_evts_times.append(f - frame_win)
                urine_xys = urine_px_to_cm(true_evts, cent_xy=self.arena_cnt, px_per_cm=self.px_per_cm)
                urine_evts_xys.append(urine_xys)

            if show_vid or save_vid is not None:
                if show_vid == 2:
                    out_frame = self.show_all_steps(mask_buf[0], fillpt_buf[0])
                else:
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
        if save_vid is not None:
            out_vid.release()
            print('Peetect video saved')
        cv2.destroyAllWindows()

        # Convert to ndarrays instead of lists
        urine_evts_xys = np.array(urine_evts_xys, dtype=np.ndarray)
        urine_data = np.array([])
        if len(urine_evts_xys) > 0:
            urine_data = expand_urine_data(urine_evts_xys, times=urine_evts_times)
        return urine_data

    def add_dz(self, zone=None, num_pts=0):
        w1 = np.array([[316, 210], [330, 210], [330, 480], [316, 480]])
        w2 = np.array([[280, 215], [101, 115], [129, 85], [306, 197]])
        w3 = np.array([[350, 215], [545, 95], [530, 70], [337, 195]])
        c_post = np.array([[337, 165], [356, 178], [368, 198], [367, 223], [356, 242], [336, 253], [311, 250],
                           [292, 238], [282, 219], [282, 193], [292, 175], [314, 166]]) + [-2, 3]
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

    def show_all_steps(self, mask_list, pnts):
        top_half = cv2.cvtColor(np.hstack(mask_list[:3]), cv2.COLOR_GRAY2BGR)
        left_half = cv2.cvtColor(np.hstack(mask_list[3:-1]), cv2.COLOR_GRAY2BGR)
        fframe = cv2.cvtColor(mask_list[-1], cv2.COLOR_GRAY2BGR)
        for s in pnts:
            slp_pnt = s.astype(int)
            cv2.circle(fframe, (slp_pnt[0], slp_pnt[1]), 3, (0, 100, 200), -1, cv2.LINE_AA)
        bot_half = np.hstack((left_half, fframe))
        concat_masks = np.vstack((top_half, bot_half))
        return concat_masks

    def show_output(self, raw_frame, urine_pnts, sleap_pnts):
        cv2.circle(raw_frame, (self.arena_cnt[0] + 12, self.arena_cnt[1] + 18), 195, (255, 255, 255, 255), 1, cv2.LINE_AA)
        urine_mask = np.zeros_like(raw_frame[:, :, 0])
        if len(urine_pnts) > 0:
            urine_mask[urine_pnts[:, 1], urine_pnts[:, 0]] = 1
            contours, heir = cv2.findContours(urine_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(raw_frame, contours, -1, (0, 255, 0), 1)
        for s in sleap_pnts:
            slp_pnt = s.astype(int)
            cv2.circle(raw_frame, (slp_pnt[0], slp_pnt[1]), 3, (0, 100, 200), -1, cv2.LINE_AA)
        for d in self.dead_zones:
            cv2.polylines(raw_frame, [d], True, (0, 0, 250), 1, cv2.LINE_AA)

        big_frame = cv2.resize(raw_frame, (1280, 960))
        font = ImageFont.truetype('arial.ttf', 48)
        img_pil = Image.fromarray(big_frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((20, 10), 'Fill Points', font=font, fill=(0, 100, 200, 0))
        draw.text((20, 60), 'Dead Zones', font=font, fill=(0, 0, 250, 0))
        draw.text((20, 110), 'Mark Detected', font=font, fill=(0, 200, 100, 0))
        img = np.array(img_pil)
        return img


def proj_urine_across_time(urine_data, thresh=0):
    all_xys = urine_data[:, 1:]
    unique_xys, cnts = np.unique(all_xys, axis=0, return_counts=True)
    unique_xys = unique_xys[cnts > thresh, :]
    return unique_xys


def urine_segmentation(urine_data, space_dist=1, time_dist=5):
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


def urine_across_time(expand_urine, len_s=0, hz=40):
    times = expand_urine[:, 0]
    if len_s == 0:
        len_s = np.max(times)
    urine_over_time = np.zeros(int(len_s*hz))
    unique_ts, urine_cnts = np.unique(times, return_counts=True)
    urine_over_time[unique_ts.astype(int)] = urine_cnts
    return urine_over_time


def dist_to_urine(x, y, expand_urine):
    xy_data = np.vstack((x, y)).T
    urine_xy_cm = proj_urine_across_time(expand_urine)
    dist_acc = []
    for xy in xy_data:
        xy_vec = np.ones((len(urine_xy_cm), 2)) * np.expand_dims(xy, 1).T
        dists = np.sqrt(np.sum((xy_vec - urine_xy_cm)**2, axis=1))
        dist_acc.append(min(dists))
    return np.array(dist_acc)
