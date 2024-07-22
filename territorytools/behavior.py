import numpy as np
from urine import urine_across_time
import warnings

# Disabling runtime warning for mean of empty slice which doesn't seem to relate to any issues
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_territory_data(sleap_pts, rot_offset=0, px_per_ft=350, ref_point=(638, 504), trunk_ind=5, head_ind=0):
    slp_cm = xy_to_cm_vec(sleap_pts, center_pt=ref_point, px_per_ft=px_per_ft)
    slp_rot = rotate_xy_vec(slp_cm, rot_offset)
    head_angles = get_head_direction(slp_rot, trunk_ind=trunk_ind, head_ind=head_ind)
    mouse_cent = np.nanmean(slp_rot, axis=1).T
    cent_x = mouse_cent[:, 0]
    cent_y = mouse_cent[:, 1]
    rel_xy = mouse_cent
    dist_vec = np.linalg.norm(rel_xy[1:, :] - rel_xy[:-1, :], axis=1) / (px_per_ft/30.48)
    dist_vec = np.hstack(([0], dist_vec))
    return cent_x, cent_y, head_angles, dist_vec, slp_rot


def get_diadic_behavior(sub_x, sub_y, sub_heading, stim_x, stim_y):
    sub_xy = np.vstack((sub_x, sub_y)).T
    stim_xy = np.vstack((stim_x, stim_y)).T
    stim_rel_xy = stim_xy - sub_xy
    dist_btw_mice = np.linalg.norm(stim_rel_xy, axis=1)
    ang_stim_to_sub = np.arctan2(stim_rel_xy[:, 1], stim_rel_xy[:, 0])
    sub_rel_ang_stim = sub_heading - ang_stim_to_sub
    return dist_btw_mice, sub_rel_ang_stim


def rotate_xy_vec(pts, rot_ang):
    out_pts = []
    for i in range(pts.shape[-1]):
        rot_pts = rotate_xy(pts[0, :, i], pts[1, :, i], rot_ang)
        out_pts.append(np.vstack(rot_pts))
    xys_rot = np.array(out_pts)
    return np.moveaxis(xys_rot, 0, 2)


def rotate_xy(x, y, rot):
    rot_dict = {'rni': 0,
                'none': 0,
                'irn': 120,
                'nir': 240}
    if type(rot) == str:
        rot_deg = rot_dict[rot]
    else:
        rot_deg = rot
    in_rad = np.radians(rot_deg)
    c, s = np.cos(in_rad), np.sin(in_rad)
    rot_mat = [[c, -s], [s, c]]
    xy = rot_mat @ np.vstack((x, y))
    return xy[0, :], xy[1, :]


def xy_to_cm(xy, center_pt=(325, 210), px_per_ft=225):
    rad_cm = 30.48  # radius of arena in cm (12in)
    px_per_cm = px_per_ft/rad_cm
    rel_xy = xy - center_pt
    rel_xy[:, 1] = -rel_xy[:, 1]
    cm_x = rel_xy[:, 0] / px_per_cm
    cm_y = rel_xy[:, 1] / px_per_cm
    return cm_x, cm_y


def xy_to_cm_vec(xys, center_pt=(325, 210), px_per_ft=225):
    xys_rot = []
    for i in range(xys.shape[-1]):
        this_xy = xys[:, :, i].T
        cm_x, cm_y = xy_to_cm(this_xy, center_pt=center_pt, px_per_ft=px_per_ft)
        xys_rot.append(np.vstack((cm_x, cm_y)))
    xys_rot = np.array(xys_rot)
    return np.moveaxis(xys_rot, 0, 2)


def get_head_direction(mouse_data, in_deg=False, trunk_ind=5, head_ind=0): #originally 4, 1
    md_copy = np.copy(mouse_data)
    md_copy[1, :, :] = -md_copy[1, :, :]
    xy_nose = md_copy[:, head_ind, :] - md_copy[:, trunk_ind, :]
    angs = np.arctan2(xy_nose[1, :], xy_nose[0, :])
    if in_deg:
        angs = np.degrees(angs)
    return angs


def compute_preferences(x, y, walls=None):
    ter_id = np.zeros_like(x)
    for i, (x0, y0) in enumerate(zip(x, y)):
        ter_id[i] = xy_to_territory(x0, y0, walls=walls)
    ter, ter_cnts = np.unique(ter_id, return_counts=True)
    prefs = np.zeros(3)
    prefs[ter.astype(int)] = ter_cnts / sum(ter_cnts)
    return prefs, ter_id


def xy_to_territory(x, y, walls=None):
    if walls is None:
        walls = [-np.pi / 2, 5 * np.pi / 6, np.pi / 6]
    t = np.arctan2(y, x)
    ter_a = np.logical_or(t < walls[0], t > walls[1])
    ter_b = np.logical_and(t > walls[0], t < walls[2])
    ter_c = np.logical_and(t > walls[2], t < walls[1])
    ter_id = np.argmax([ter_a, ter_b, ter_c])
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
    #Get average angle of head direction data
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


def make_design_matrix(run_data, md):
    dist_btw_mice, looking_ang = [], []
    if md['block'] == '0':
        dist_btw_mice, looking_ang = get_diadic_behavior(run_data[0]['x_cm'], run_data[0]['y_cm'], run_data[0]['angle'],
                                                         run_data[1]['x_cm'], run_data[1]['y_cm'])
    design_mat = []
    for m, d in enumerate(run_data):
        ut = urine_across_time(d['urine_data'], len_s=len(d['x_cm'])/40)
        ut[ut > 1000] = np.nan
        # ut = (ut > 0).astype(int)
        vel = d['velocity'] * 40
        b_feats = [d['x_cm'], d['y_cm'], vel, d['angle'], ut]
        for ft in b_feats:
            norm_f = (ft - np.nanmin(ft)) / (np.nanmax(ft) - np.nanmin(ft))
            norm_f = ft
            design_mat.append(norm_f)

    if len(dist_btw_mice) > 0:
        fts = [dist_btw_mice, looking_ang]
        for f in fts:
            norm_f = (f - np.nanmin(f)) / (np.nanmax(f) - np.nanmin(f))
            norm_f = f
            design_mat.append(norm_f)

    design_mat = np.array(design_mat)

    return design_mat
