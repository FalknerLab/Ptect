from territorytools.urine import urine_across_time
from territorytools.utils import xy_to_cm_vec, rotate_xy_vec
import numpy as np
from scipy.stats import zscore
import warnings

# Disabling runtime warning for mean of empty slice which doesn't seem to relate to any issues
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_territory_data(sleap_pts, rot_offset=0, px_per_cm=350/30.48, ref_point=(638, 504), hz=40, trunk_ind=5, head_ind=0):
    slp_cm = xy_to_cm_vec(sleap_pts, center_pt=ref_point, px_per_cm=px_per_cm)
    slp_rot = rotate_xy_vec(slp_cm, rot_offset)
    head_angles = get_head_direction(slp_rot, trunk_ind=trunk_ind, head_ind=head_ind)
    mouse_cent = np.nanmean(slp_rot, axis=1).T
    cent_x = mouse_cent[:, 0]
    cent_y = mouse_cent[:, 1]
    rel_xy = mouse_cent
    dist_vec = np.linalg.norm(rel_xy[1:, :] - rel_xy[:-1, :], axis=1) / px_per_cm
    dist_vec = np.hstack(([0], dist_vec)) * hz
    return cent_x, cent_y, head_angles, dist_vec, slp_rot


def get_diadic_behavior(sub_x, sub_y, sub_heading, stim_x, stim_y):
    sub_xy = np.vstack((sub_x, sub_y)).T
    stim_xy = np.vstack((stim_x, stim_y)).T
    stim_rel_xy = stim_xy - sub_xy
    dist_btw_mice = np.linalg.norm(stim_rel_xy, axis=1)
    ang_stim_to_sub = np.arctan2(stim_rel_xy[:, 1], stim_rel_xy[:, 0])
    sub_rel_ang_stim = sub_heading - ang_stim_to_sub
    return dist_btw_mice, sub_rel_ang_stim


def get_head_direction(mouse_data, in_deg=False, trunk_ind=5, head_ind=0): #originally 4, 1
    md_copy = np.copy(mouse_data)
    md_copy[1, :, :] = -md_copy[1, :, :]
    xy_nose = md_copy[:, head_ind, :] - md_copy[:, trunk_ind, :]
    angs = np.arctan2(xy_nose[1, :], xy_nose[0, :])
    if in_deg:
        angs = np.degrees(angs)
    return angs


def compute_preferences(x, y, walls=None):
    ter_id = xy_to_territory(x, y, walls=walls)
    ter, ter_cnts = np.unique(ter_id, return_counts=True)
    prefs = np.zeros(3)
    prefs[ter.astype(int)] = ter_cnts / sum(ter_cnts)
    return prefs, ter_id


def prefs_across_time(x, y, samp_win=5400):
    pref_acc = []
    for i in range(len(x)-samp_win):
        pref_acc.append(compute_preferences(x[i:i+samp_win], y[i:i+samp_win])[0])
    return np.array(pref_acc)


def vel_across_time(vel, ter_id, samp_win=5400):
    mean_vel_self = []
    mean_vel_other = []
    mean_vel_novel = []
    for i in range(len(vel)-samp_win):
        this_vel = vel[i:i+samp_win]
        this_ter = ter_id[i:i+samp_win]
        for ind, acc in enumerate((mean_vel_self, mean_vel_other, mean_vel_novel)):
            acc.append(np.nanmean(this_vel[this_ter == ind]))
    return mean_vel_self, mean_vel_other, mean_vel_novel


def xy_to_territory(x, y, walls=None):
    if walls is None:
        walls = [-np.pi / 2, 5 * np.pi / 6, np.pi / 6]
    t = np.arctan2(y, x)
    ter_a = np.logical_or(t < walls[0], t > walls[1])
    ter_b = np.logical_and(t > walls[0], t < walls[2])
    ter_c = np.logical_and(t > walls[2], t < walls[1])
    ter_id = np.argmax(np.vstack([ter_a, ter_b, ter_c]), axis=0)
    return ter_id


def interp_behavs(*args):
    out_behavs = []
    for b in args:
        interp_b = b.copy()
        nans = np.isnan(interp_b)
        interp_vals = np.interp(nans.nonzero()[0], (~nans).nonzero()[0], interp_b[~nans])
        interp_b[nans] = interp_vals
        out_behavs.append(interp_b)
    return out_behavs


def avg_angs(head_angs):
    #Get average angle of head direction dataset
    avg_s = np.nanmean(np.sin(head_angs))
    avg_c = np.nanmean(np.cos(head_angs))
    return np.arctan2(avg_s, avg_c)


def xy_to_polar(x, y):
    t = np.arctan2(y, x)
    xy = np.vstack((x[:, None], y[:, None])).T
    r = np.linalg.norm(xy, axis=1)
    return t, r


def compute_over_spatial_bin(xpos, ypos, data, func, bins=20, range=None):
    if range is None:
        range = [[-32, 32], [-32, 32]]
    _, xedges, yedges = np.histogram2d(xpos, ypos, bins=bins, range=range)
    out_hist = np.zeros((len(xedges) - 1, len(yedges) - 1))
    for i, xe in enumerate(xedges[:-1]):
        in_x = np.logical_and(xpos > xe, xpos < xedges[i + 1])
        for j, ye in enumerate(yedges[:-1]):
            in_y = np.logical_and(ypos > ye, ypos < yedges[j + 1])
            in_bin = np.logical_and(in_x, in_y)
            out_hist[i, j] = func(data[in_bin])
    return out_hist, xedges, yedges


def make_design_matrix(run_data, md, norm=None):
    dist_btw_mice, looking_ang = [], []
    if md['Territory']['block'] == '0':
        dist_btw_mice, looking_ang = get_diadic_behavior(run_data[0]['x_cm'], run_data[0]['y_cm'], run_data[0]['angle'],
                                                         run_data[1]['x_cm'], run_data[1]['y_cm'])
    design_mat = []
    for m, d in enumerate(run_data):
        x = np.abs(d['x_cm'])
        ut = urine_across_time(d['urine_data'], len_s=len(d['x_cm'])/40)
        vel = d['velocity'] * 40
        b_feats = [x, d['y_cm'], vel, d['angle'], ut]
        for ft in b_feats:
            norm_f = ft
            if norm == '0-1':
                norm_f = (ft - np.nanmin(ft)) / (np.nanmax(ft) - np.nanmin(ft))
            elif norm == 'zscore':
                norm_f = zscore(ft)
            design_mat.append(norm_f)

    if len(dist_btw_mice) > 0:
        fts = [dist_btw_mice, looking_ang]
        for f in fts:
            norm_f = f
            if norm == '0-1':
                norm_f = (f - np.nanmin(f)) / (np.nanmax(f) - np.nanmin(f))
            elif norm == 'zscore':
                norm_f = zscore(f)
            design_mat.append(norm_f)

    design_mat = np.array(design_mat).T

    return design_mat


def prob_over_x(x_data, binary_data, bin_min, bin_max, bin_num=20):
    bins = np.linspace(bin_min, bin_max, bin_num)
    tot_binary = np.sum(binary_data)
    probs = []
    prior = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        this_bin = np.logical_and(x_data > b0, x_data < b1)
        probs.append(np.sum(binary_data[this_bin]) / tot_binary)
        prior.append(np.sum(this_bin)/len(x_data))
    prior = 100*np.array(prior)
    probs = 100*np.array(probs)
    bin_left_edge = bins[:-1]
    return probs, prior, bin_left_edge


def time_near_walls(x_pos, y_pos, wall_angs, dist_thresh=2, rad=30.48):
    percents = []
    in_zone = []
    for a in wall_angs:
        dtw = dist_to_wall(x_pos, y_pos, a)
        percents.append(100 * (sum(dtw < dist_thresh) / len(x_pos)))
        in_zone.append(dtw < dist_thresh)
    return percents, in_zone


def dist_to_wall(x, y, wall_deg, rad=30.48, num_pts=1000):
    xy_data = np.vstack((x, y)).T
    dist_acc = []
    wall_xs = np.linspace(0, rad * np.sin(np.radians(wall_deg)), num_pts)
    wall_ys = np.linspace(0, rad * np.cos(np.radians(wall_deg)), num_pts)
    wall_pts = np.vstack((wall_xs, wall_ys)).T
    for xy in xy_data:
        xy_vec = np.ones((len(wall_xs), 2)) * np.expand_dims(xy, 1).T
        dists = np.sqrt(np.sum((xy_vec - wall_pts)**2, axis=1))
        dist_acc.append(np.min(dists))
    return np.array(dist_acc)
