import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from PIL import ImageFont, ImageDraw, Image
from territorytools.behavior import xy_to_cm


def sleap_to_fill_pts(sleap_h5):
    with h5py.File(sleap_h5, "r") as f:
        locations = f["tracks"][:].T
    t, d1, d2, num_mice = np.shape(locations)
    fill_pts = []
    last_pts = np.empty(num_mice, dtype=np.ndarray)
    for i in range(t):
        for m in range(num_mice):
            t_pts = locations[i, :, :, m]
            t_pts = np.moveaxis(t_pts, 0, 1)
            keep = ~np.all(np.isnan(t_pts), axis=0)
            if np.any(keep):
                k_pts = t_pts[:, keep].T
                last_pts[m] = k_pts
        fill_pts.append(np.vstack(last_pts))
    return fill_pts


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
                 hot_thresh=70, cold_thresh=30, s_kern=5, di_kern=5, v_mask=None):
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
        self.heat_thresh = hot_thresh
        self.cool_thresh = cold_thresh
        self.smooth_kern = s_kern
        self.dilate_kern = di_kern
        if v_mask is None:
            self.valid_zone = cv2.circle(np.zeros((height, width)), (self.arena_cnt[0], self.arena_cnt[1]),
                                    int(px_per_cm*30.48), 255, -1)
            self.valid_zone = self.valid_zone.astype('uint8')
        else:
            self.valid_zone = v_mask

    def peetect_frames(self, start_frame=0, num_frames=None, save_vid=None, show_vid=False,
                       hz=40, cool_thresh=None, hot_thresh=None):

        if hot_thresh is None:
            hot_thresh = self.heat_thresh

        if cool_thresh is None:
            cool_thresh = self.cool_thresh

        out_vid = None
        if save_vid is not None:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out_vid = cv2.VideoWriter(save_vid, fourcc, 120, (1280, 960), isColor=True)

        # setup video object
        vid_obj = cv2.VideoCapture(self.thermal_vid)

        # if not specified, peetect whole video
        if num_frames is None:
            num_frames = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))

        # offset video and fill points to start frame
        vid_obj.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fill_pnts = self.fill_pts[start_frame:]

        # collect frame times with urine and urine xys
        urine_evts_times = []
        urine_evts_xys = []
        cool_evts_xys = np.empty((0, 2))
        print('Running Peetect...')
        for f in range(0, num_frames):
            if f % 100 == 0:
                print('Running Peetect on frame: ', f, ' of ', num_frames)

            # run detection on next frame
            is_read, frame_i = vid_obj.read()

            if is_read:
                this_evts, cool_evts, mask = self.peetect(frame_i, fill_pnts[f], cool_thresh=cool_thresh, hot_thresh=hot_thresh)

                # if good urine detected, convert to cm and add to output
                if len(this_evts) > 0:
                    urine_evts_times.append(f / hz)
                    urine_evts_xys.append(this_evts)

                if len(cool_evts) > 0:
                    cool_evts_xys = np.vstack((cool_evts_xys, cool_evts))
                    cool_evts_xys = np.unique(cool_evts_xys, axis=0)

                out_frame = []
                if show_vid or save_vid is not None:
                    if show_vid == 2:
                        out_frame = self.show_all_steps(mask, fill_pnts[f])
                    else:
                        out_frame = self.show_output(frame_i, this_evts, fill_pnts[f], cool_evts)

                if save_vid is not None:
                    out_vid.write(out_frame)

                if show_vid:
                    cv2.imshow('Peetect Output', out_frame)
                    cv2.waitKey(1)

        if save_vid is not None:
            out_vid.release()
            print('Peetect video saved')
        cv2.destroyAllWindows()

        true_evts = []
        true_ts = []
        cool_evts_1d = list([''.join(str(row)) for row in cool_evts_xys.astype(int)])
        for i, (t, t_e) in enumerate(zip(urine_evts_times, urine_evts_xys)):
            print(fr'Post-fix event {i} of {len(urine_evts_times)}')
            te_1d = list([''.join(str(row)) for row in t_e])
            valid_urine, te_inds, ce_inds = np.intersect1d(te_1d, cool_evts_1d, return_indices=True)
            if len(valid_urine) > 0:
                true_evts.append(t_e[te_inds, :])
                true_ts.append(t)

        # for i, (t, t_e) in enumerate(zip(urine_evts_times, urine_evts_xys)):
        #     print(fr'Post-fix event {i} of {len(urine_evts_times)}')
        #     this_te = []
        #     for e in t_e:
        #         if list(e) in cool_evts_xys.tolist():
        #             this_te.append(e)
        #     if len(this_te) > 0:
        #         true_evts.append(np.array(this_te))
        #         true_ts.append(t)

        urine_data = np.array([])
        if len(true_evts) > 0:
            true_evts_cm = self.urine_px_to_cm(true_evts)
            urine_data = expand_urine_data(true_evts_cm, times=true_ts)
        print('Peetect Finished')

        return urine_data

    def urine_px_to_cm(self, pts_list):
        pts_cm = []
        for pts in pts_list:
            x = (pts[:, 0] - self.arena_cnt[0]) / self.px_per_cm
            y = -(pts[:, 1] - self.arena_cnt[1]) / self.px_per_cm
            pts_cm.append(np.vstack((x, y)).T)
        return pts_cm

    def urine_px_to_cm_2d(self, pts):
        x = (pts[:, 0] - self.arena_cnt[0]) / self.px_per_cm
        y = -(pts[:, 1] - self.arena_cnt[1]) / self.px_per_cm
        pts_cm = np.vstack((x, y)).T
        return pts_cm

    def peetect(self, frame, pts, hot_thresh=70, cool_thresh=30):
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
        heat_mask = np.uint8(255 * (dz_frame > hot_thresh))
        test = heat_mask.copy()

        # dilate resulting mask to hopefully merge mouse parts and expand urine
        di_frame = self.dilate_frame(heat_mask)
        # di_frame = heat_mask

        # fill in all the given points with black
        fill_frame = self.fill_frame_with_points(di_frame, pts, im_w, im_h)

        # if urine detected set output to urine indices
        urine_xys = []
        if np.sum(fill_frame) > 0:
            urine_xys = np.argwhere(fill_frame > 0)

        # mask valid zone but white
        valid_frame = self.mask_valid_zone(frame_smooth, fill='w')

        # whiten deadzones
        dz_frame_w = self.fill_deadzones(valid_frame, fill='w')

        cool_mask = dz_frame_w < cool_thresh

        # cool xys
        cool_xys = []
        if np.sum(cool_mask) > 0:
            cool_xys = np.argwhere(cool_mask > 0)

        return urine_xys, cool_xys, [f1, frame_smooth, dz_frame, test, di_frame, fill_frame]

    def fill_deadzones(self, frame, fill=None):
        c = (0, 0, 0)
        if fill == 'w':
            c = (255, 255, 255)
        for dz in self.dead_zones:
            dz2 = np.array(dz).astype(int)
            cv2.fillPoly(frame, pts=[dz2], color=c)
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

    def mask_valid_zone(self, frame, fill=None):
        valid_frame = cv2.bitwise_and(frame, frame, mask=self.valid_zone)
        if fill == 'w':
            cv2.floodFill(valid_frame, None, (1, 1), 255)
        return valid_frame

    def fill_frame_with_points(self, frame, pnts, width, height):
        cop_f = frame.copy()
        for p in pnts:
            px = int(p[0])
            py = int(p[1])
            if 0 < px < width and 0 < py < height:
                if frame[py, px] > 0:
                    cv2.floodFill(cop_f, None, (px, py), 0)
        return cop_f

    def add_dz(self, zone=None, num_pts=0):
        w1 = np.array([[316, 210], [330, 210], [330, 480], [316, 480]])
        w2 = np.array([[280, 215], [110, 118], [129, 100], [306, 197]])
        w3 = np.array([[350, 215], [545, 95], [530, 70], [337, 195]]) + [5, 5]
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
        concat_masks = np.zeros(640, 480)
        if len(mask_list) > 0:
            top_half = cv2.cvtColor(np.hstack(mask_list[:3]), cv2.COLOR_GRAY2BGR)
            left_half = cv2.cvtColor(np.hstack(mask_list[3:-1]), cv2.COLOR_GRAY2BGR)
            fframe = cv2.cvtColor(mask_list[-1], cv2.COLOR_GRAY2BGR)
            for s in pnts:
                slp_pnt = s.astype(int)
                cv2.circle(fframe, (slp_pnt[0], slp_pnt[1]), 3, (0, 100, 200), -1, cv2.LINE_AA)
            bot_half = np.hstack((left_half, fframe))
            concat_masks = np.vstack((top_half, bot_half))
        return concat_masks

    def show_output(self, raw_frame, urine_pnts, sleap_pnts, cool_pnts):

        cv2.circle(raw_frame, (self.arena_cnt[0], self.arena_cnt[1]), 200, (255, 255, 255, 255), 1, cv2.LINE_AA)
        cols = ((0, 1), (1, 2))
        for pnts, c in zip((urine_pnts, cool_pnts), cols):
            if len(pnts) > 0:
                urine_mask = np.zeros_like(raw_frame)
                raw_frame[pnts[:, 0], pnts[:, 1], c[0]] = 0
                raw_frame[pnts[:, 0], pnts[:, 1], c[1]] = 0
                # urine_mask[urine_mask == 0] = np.nan
                # contours, heir = cv2.findContours(urine_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # cv2.drawContours(raw_frame, contours, -1, c, 1)
                # cv2.addWeighted(raw_frame, 1, urine_mask, 0.5)

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

    def proj_cool(self, show_vid=True, start_frame=0):

        # setup video object
        vid_obj = cv2.VideoCapture(self.thermal_vid)

        if start_frame > 0:
            vid_obj.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        num_frames = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        out_xys = np.empty((0, 2))
        base_mask = None
        for i in range(num_frames - start_frame):
            if i % 1000 == 0:
                print(f'Frame {i} of {num_frames}')
            is_read, frame = vid_obj.read()
            if is_read:
                # get frame data, convert to grey
                f1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # smooth frame
                frame_smooth = self.smooth_frame(f1)

                # mask valid zone but white
                valid_frame = self.mask_valid_zone(frame_smooth, fill='w')

                # whiten deadzones
                dz_frame_w = self.fill_deadzones(valid_frame, fill='w')

                cool_mask = dz_frame_w < self.cool_thresh

                # cool xys
                if np.sum(cool_mask) > 0:
                    cool_xys = np.argwhere(cool_mask > 0)
                    out_xys = np.unique(np.vstack([out_xys, cool_xys]), axis=0).astype(int)
                    if base_mask is None:
                        b_mask = dz_frame_w < (self.cool_thresh * 1.1)
                        base_mask = np.argwhere(b_mask > 0)



                if show_vid:
                    out_frame = self.show_output(frame, out_xys, [], [])
                    cv2.imshow('Peetect Output', out_frame)
                    cv2.waitKey(1)

        base_m_1d = list([''.join(str(row)) for row in base_mask])
        p_data_1d = list([''.join(str(row)) for row in out_xys])
        valid_urine, _, p_data_inds = np.intersect1d(base_m_1d, p_data_1d, return_indices=True)
        out_xys = np.delete(out_xys, p_data_inds, 0)
        base_mask = xy_to_cm(np.fliplr(base_mask), center_pt=self.arena_cnt, px_per_ft=self.px_per_cm*30.48)
        out_xys = xy_to_cm(np.fliplr(out_xys), center_pt=self.arena_cnt, px_per_ft=self.px_per_cm * 30.48)
        return np.array(out_xys).T, np.array(base_mask).T

        # out_x, out_y = xy_to_cm(out_xys, center_pt=self.arena_cnt, px_per_ft=self.px_per_cm*30.48)
        # base_x, base_y = xy_to_cm(base_mask)
        # return np.vstack((out_x, out_y)).T, np.vstack((base_x, base_y)).T

        # out_xys = self.urine_px_to_cm_2d(out_xys)
        # base_xys = self.urine_px_to_cm_2d(base_mask)
        # return out_xys, base_xys


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


def dist_to_urine(x, y, expand_urine, thresh=0):
    xy_data = np.vstack((x, y)).T
    urine_xy_cm = proj_urine_across_time(expand_urine, thresh=thresh)
    dist_acc = []
    for xy in xy_data:
        xy_vec = np.ones((len(urine_xy_cm), 2)) * np.expand_dims(xy, 1).T
        dists = np.sqrt(np.sum((xy_vec - urine_xy_cm)**2, axis=1))
        dist_acc.append(np.min(dists))
    return np.array(dist_acc)

