import time
from abc import abstractmethod, ABC
from types import NoneType
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from PIL import ImageFont, ImageDraw, Image
from territorytools.utils import xy_to_cm, rotate_xy, intersect2d


def sleap_to_fill_pts(sleap_h5):
    with h5py.File(sleap_h5, "r") as f:
        locations = f["tracks"][:].T
    t, d1, d2, num_mice = np.shape(locations)
    fill_pts = []
    last_pts = np.empty(num_mice * d1, dtype=np.ndarray)
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

class PtectPipe(ABC):
    buffer=[]
    @abstractmethod
    def send(self, *args):
        pass

    def read(self):
        return self.buffer

class Peetector:
    def __init__(self, avi_file, flood_pnts, dead_zones=[], cent_xy=(320, 212), px_per_cm=7.38188976378, check_frames=1,
                 hot_thresh=70, cold_thresh=30, s_kern=5, di_kern=5, hz=40, v_mask=None, frame_type=None, radius=30,
                 rot_ang=0, start_frame=0):
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
        self.radius = radius
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


    def get_length(self):
        return self.vid_obj.get(cv2.CAP_PROP_FRAME_COUNT)

    def read_frame(self):
        ret, frame_i = self.vid_obj.read()
        return ret, frame_i

    def set_frame(self, frame_num):
        self.vid_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.current_frame = frame_num

    def set_time_win(self, check_frames):
        if check_frames != self.time_thresh:
            self.time_thresh = check_frames
            self.buffer = PBuffer(self.time_thresh)

    def run_ptect(self, pipe: PtectPipe=None, start_frame=0, end_frame=0, save_path=None, verbose=False):
        self.buffer = PBuffer(self.time_thresh)
        self.set_frame(start_frame)
        if end_frame <= 0:
            end_frame = self.total_frames

        all_hot_data = np.empty((0, 3))
        all_cool_data = np.empty((0, 3))

        frame_c = 0
        tot_frames = end_frame - start_frame
        # t0 = time.time()
        while self.current_frame < end_frame:
            if verbose and frame_c % 1000 == 0:
                print(f'Running Ptect on frame {frame_c} of {tot_frames}...')

            if pipe is not None:
                pipe.send((frame_c, tot_frames))
                is_done = pipe.read()
                if is_done:
                    return None

            p_out = self.peetect_next_frame()[0]

            all_hot_data = np.vstack((all_hot_data, p_out[0]))
            all_cool_data = np.vstack((all_cool_data, p_out[1]))

            frame_c += 1
            # t1 = time.time()
            # print(t1-t0)
            # t0 = t1

        hot_half = np.hstack((all_hot_data, np.ones_like(all_hot_data[:, 0][:, None])))
        cool_half = np.hstack((all_cool_data, np.zeros_like(all_cool_data[:, 0][:, None])))
        all_data = np.vstack((hot_half, cool_half))

        if save_path is not None:
            np.savez(save_path, urine_data=all_data)
        else:
            return all_hot_data, all_cool_data


    def run_ptect_video(self, start_frame=None, num_frames=None, save_vid=None, show_vid=False, verbose=False):

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

        hot_thresh = self.heat_thresh
        cool_thresh = self.cool_thresh
        rot_ang = self.rot_ang
        fill_pnts = self.fill_pts

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
                    out_frame = self.show_all_steps(mask, fill_pnts[self.current_frame])
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

        return urine_xys, cool_xys, [f1, frame_smooth, dz_frame, test, di_frame, fill_frame, cool_mask]


    def set_valid_arena(self, shape, *args):
        self.valid_zone = make_shape_mask(self.width, self.height, shape,
                                          self.arena_cnt[0], self.arena_cnt[1],*args)
        if shape == 'circle':
            self.radius = args[0]

    def urine_px_to_cm_rot(self, pts_list, rot_ang=0):
        pts_cm = []
        for pts in pts_list:
            x = (pts[:, 1] - self.arena_cnt[0]) / self.px_per_cm
            y = -(pts[:, 0] - self.arena_cnt[1]) / self.px_per_cm
            # xy = np.vstack((x, y)).T
            x, y = rotate_xy(x, y, rot_ang)
            pts_cm.append(np.vstack((x, y)).T)
        return pts_cm

    def urine_px_to_cm_2d(self, pts):
        x = (pts[:, 0] - self.arena_cnt[0]) / self.px_per_cm
        y = -(pts[:, 1] - self.arena_cnt[1]) / self.px_per_cm
        pts_cm = np.vstack((x, y)).T
        return pts_cm

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

    def show_all_steps(self, mask_list, pnts):
        concat_masks = np.zeros((640, 480))
        if len(mask_list) > 0:
            top_half = cv2.cvtColor(np.hstack(mask_list[:3]), cv2.COLOR_GRAY2BGR)
            left_half = cv2.cvtColor(np.hstack(mask_list[3:-2]), cv2.COLOR_GRAY2BGR)
            fframe = cv2.cvtColor(mask_list[0], cv2.COLOR_GRAY2BGR)
            fframe[mask_list[-1] == 1] = [255, 255, 0]
            for s in pnts:
                slp_pnt = s.astype(int)
                cv2.circle(fframe, (slp_pnt[0], slp_pnt[1]), 3, (100, 100, 200), -1, cv2.LINE_AA)
            bot_half = np.hstack((left_half, fframe))
            concat_masks = np.vstack((top_half, bot_half))
        return concat_masks

    def show_output(self, raw_frame, urine_pnts, sleap_pnts, cool_pnts):

        cv2.circle(raw_frame, (self.arena_cnt[0], self.arena_cnt[1]), self.radius, (255, 255, 255, 255), 1, cv2.LINE_AA)
        cols = ((0, 1), (1, 2))
        for pnts, c in zip((cool_pnts, urine_pnts), cols):
            if len(pnts) > 0:
                raw_frame[pnts[:, 0], pnts[:, 1], c[0]] = 255
                raw_frame[pnts[:, 0], pnts[:, 1], c[1]] = 255

        for s in sleap_pnts:
            slp_pnt = s.astype(int)
            cv2.circle(raw_frame, (slp_pnt[0], slp_pnt[1]), 3, (100, 100, 200), -1, cv2.LINE_AA)
        for d in self.dead_zones:
            pts = np.array(d, dtype=np.int32)
            cv2.polylines(raw_frame, [pts], True, (0, 0, 250), 1, cv2.LINE_AA)

        big_frame = cv2.resize(raw_frame, (1280, 960))
        font = ImageFont.truetype('arial.ttf', 48)
        img_pil = Image.fromarray(big_frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((20, 10), 'Fill Points', font=font, fill=(100, 100, 200, 0))
        draw.text((20, 60), 'Dead Zones', font=font, fill=(0, 0, 250, 0))
        draw.text((20, 110), 'Cool Mark', font=font, fill=(255, 255, 0, 0))
        draw.text((20, 160), 'Hot Mark', font=font, fill=(0, 255, 255, 0))
        img = np.array(img_pil)
        return img


class PBuffer:
    def __init__(self, buffer_size):
        buffer_size = max(buffer_size, 1)
        self.size = buffer_size
        self.buffer = np.empty(buffer_size, dtype=np.ndarray)
        self.pos = 0

    def push(self, data):
        self.buffer[self.pos] = data
        self.pos += 1
        if self.pos >= self.size:
            self.pos = 0

    def pop(self):
        pos = self.pos - 1
        if pos < 0:
            pos = self.size - 1
        return self.buffer[pos]

    def no_empty(self):
        any_none = np.vectorize(type)(self.buffer)
        any_none = np.any(any_none == NoneType)
        if any_none:
            return False
        lens = np.vectorize(np.size)(self.buffer)
        no_empty = np.all(lens)
        return no_empty

    def check_ahead(self):
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
    all_xys = urine_data[:, 1:]
    unique_xys, unique_indices, cnts = np.unique(all_xys, axis=0, return_index=True, return_counts=True)
    unique_xys = unique_xys[cnts > thresh, :]
    return unique_xys, all_xys[unique_indices, 0]


def split_urine_data(urine_data):
    hot_ind = urine_data[:, 3].astype(bool)
    return urine_data[hot_ind, :3], urine_data[~hot_ind, :3]


def make_mark_raster(urine_data, hot_thresh=0, cool_thresh=0):
    hot_data, cool_data = split_urine_data(urine_data)
    hot_rast = urine_across_time(hot_data, hot_thresh)[:, 0]
    cool_rast = urine_across_time(cool_data, cool_thresh)[:, 1]
    return hot_rast, cool_rast


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


def dist_to_urine(x, y, expand_urine, thresh=0):
    xy_data = np.vstack((x, y)).T
    urine_xy_cm = proj_urine_across_time(expand_urine, thresh=thresh)
    dist_acc = []
    for xy in xy_data:
        xy_vec = np.ones((len(urine_xy_cm), 2)) * np.expand_dims(xy, 1).T
        dists = np.sqrt(np.sum((xy_vec - urine_xy_cm)**2, axis=1))
        dist_acc.append(np.min(dists))
    return np.array(dist_acc)
