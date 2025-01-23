from territorytools.urine import urine_across_time
from territorytools.utils import xy_to_cm_vec, rotate_xy_vec
import numpy as np
from scipy.stats import zscore
import warnings

# Disabling runtime warning for mean of empty slice which doesn't seem to relate to any issues
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_territory_data(sleap_pts, rot_offset=0, px_per_cm=350/30.48, ref_point=(638, 504), hz=40, trunk_ind=5, head_ind=0):
    """
    Extracts and processes territory data from SLEAP points.

    Parameters
    ----------
    sleap_pts : numpy.ndarray
        Array of SLEAP points.
    rot_offset : float, optional
        Rotation offset in degrees (default is 0).
    px_per_cm : float, optional
        Pixels per centimeter (default is 350/30.48).
    ref_point : tuple, optional
        Reference point for centering (default is (638, 504)).
    hz : int, optional
        Sampling frequency in Hz (default is 40).
    trunk_ind : int, optional
        Index of the trunk point (default is 5).
    head_ind : int, optional
        Index of the head point (default is 0).

    Returns
    -------
    tuple
        Processed territory data including cent_x, cent_y, head_angles, dist_vec, and slp_rot.
    """
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
    """
    Computes diadic behavior metrics between two subjects.

    Parameters
    ----------
    sub_x : numpy.ndarray
        X coordinates of the subject.
    sub_y : numpy.ndarray
        Y coordinates of the subject.
    sub_heading : numpy.ndarray
        Heading angles of the subject.
    stim_x : numpy.ndarray
        X coordinates of the stimulus.
    stim_y : numpy.ndarray
        Y coordinates of the stimulus.

    Returns
    -------
    tuple
        Distance between mice and relative angle of the subject to the stimulus.
    """
    sub_xy = np.vstack((sub_x, sub_y)).T
    stim_xy = np.vstack((stim_x, stim_y)).T
    stim_rel_xy = stim_xy - sub_xy
    dist_btw_mice = np.linalg.norm(stim_rel_xy, axis=1)
    ang_stim_to_sub = np.arctan2(stim_rel_xy[:, 1], stim_rel_xy[:, 0])
    sub_rel_ang_stim = sub_heading - ang_stim_to_sub
    return dist_btw_mice, sub_rel_ang_stim


def get_head_direction(mouse_data, in_deg=False, trunk_ind=5, head_ind=0): #originally 4, 1
    """
    Computes the head direction of the mouse.

    Parameters
    ----------
    mouse_data : numpy.ndarray
        Array of mouse data points.
    in_deg : bool, optional
        Whether to return angles in degrees (default is False).
    trunk_ind : int, optional
        Index of the trunk point (default is 5).
    head_ind : int, optional
        Index of the head point (default is 0).

    Returns
    -------
    numpy.ndarray
        Array of head direction angles.
    """
    md_copy = np.copy(mouse_data)
    md_copy[1, :, :] = -md_copy[1, :, :]
    xy_nose = md_copy[:, head_ind, :] - md_copy[:, trunk_ind, :]
    angs = np.arctan2(xy_nose[1, :], xy_nose[0, :])
    if in_deg:
        angs = np.degrees(angs)
    return angs


def compute_preferences(x, y, walls=None):
    """
    Computes territory preferences based on position data.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates.
    y : numpy.ndarray
        Y coordinates.
    walls : list of float, optional
        List of wall angles (default is None).

    Returns
    -------
    tuple
        Preferences and territory IDs per sample in time.
    """
    ter_id = xy_to_territory(x, y, walls=walls)
    ter, ter_cnts = np.unique(ter_id, return_counts=True)
    prefs = np.zeros(3)
    prefs[ter.astype(int)] = ter_cnts / sum(ter_cnts)
    return prefs, ter_id


def prefs_across_time(x, y, samp_win=5400):
    """
    Computes preferences across time.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates.
    y : numpy.ndarray
        Y coordinates.
    samp_win : int, optional
        Sampling window size (default is 5400).

    Returns
    -------
    numpy.ndarray
        Array of preferences across time.
    """
    pref_acc = []
    for i in range(len(x)-samp_win):
        pref_acc.append(compute_preferences(x[i:i+samp_win], y[i:i+samp_win])[0])
    return np.array(pref_acc)


def vel_across_time(vel, ter_id, samp_win=5400):
    """
    Computes velocity across time for different territories.

    Parameters
    ----------
    vel : numpy.ndarray
        Velocity data.
    ter_id : numpy.ndarray
        Territory IDs.
    samp_win : int, optional
        Sampling window size (default is 5400).

    Returns
    -------
    tuple
        Mean velocities for self, other, and novel territories.
    """
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
    """
    Converts XY coordinates to territory IDs.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates.
    y : numpy.ndarray
        Y coordinates.
    walls : list of float, optional
        List of wall angles (default is None).

    Returns
    -------
    numpy.ndarray
        Array of territory IDs.
    """
    if walls is None:
        walls = [-np.pi / 2, 5 * np.pi / 6, np.pi / 6]
    t = np.arctan2(y, x)
    ter_a = np.logical_or(t < walls[0], t > walls[1])
    ter_b = np.logical_and(t > walls[0], t < walls[2])
    ter_c = np.logical_and(t > walls[2], t < walls[1])
    ter_id = np.argmax(np.vstack([ter_a, ter_b, ter_c]), axis=0)
    return ter_id


def interp_behavs(*args):
    """
    Interpolates behavior data to fill NaN values.

    Parameters
    ----------
    *args : tuple
        Behavior data arrays.

    Returns
    -------
    list of numpy.ndarray
        List of interpolated behavior data arrays.
    """
    out_behavs = []
    for b in args:
        interp_b = b.copy()
        nans = np.isnan(interp_b)
        interp_vals = np.interp(nans.nonzero()[0], (~nans).nonzero()[0], interp_b[~nans])
        interp_b[nans] = interp_vals
        out_behavs.append(interp_b)
    return out_behavs


def avg_angs(head_angs):
    """
    Computes the average angle of head direction data.

    Parameters
    ----------
    head_angs : numpy.ndarray
        Array of head direction angles.

    Returns
    -------
    float
        Average angle.
    """
    avg_s = np.nanmean(np.sin(head_angs))
    avg_c = np.nanmean(np.cos(head_angs))
    return np.arctan2(avg_s, avg_c)


def xy_to_polar(x, y):
    """
    Converts XY coordinates to polar coordinates.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates.
    y : numpy.ndarray
        Y coordinates.

    Returns
    -------
    tuple
        Polar coordinates (theta, radius).
    """
    t = np.arctan2(y, x)
    xy = np.vstack((x[:, None], y[:, None])).T
    r = np.linalg.norm(xy, axis=1)
    return t, r


def compute_over_spatial_bin(xpos, ypos, data, func, bins=20, range=None):
    """
    Computes a function over spatial bins.

    Parameters
    ----------
    xpos : numpy.ndarray
        X positions.
    ypos : numpy.ndarray
        Y positions.
    data : numpy.ndarray
        Data to compute over.
    func : callable
        Function to apply to each bin.
    bins : int, optional
        Number of bins (default is 20).
    range : list of list of float, optional
        Range for the bins (default is None).

    Returns
    -------
    tuple
        Histogram of computed values, x edges, and y edges.
    """
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
    """
    Creates a design matrix from run data.

    Parameters
    ----------
    run_data : list of dict
        List of dictionaries containing run data for each mouse.
    md : dict
        Metadata associated with the run.
    norm : str, optional
        Normalization method (default is None).

    Returns
    -------
    numpy.ndarray
        Design matrix.
    """
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
    """
    Computes probabilities over X data.

    Parameters
    ----------
    x_data : numpy.ndarray
        X data.
    binary_data : numpy.ndarray
        Binary data.
    bin_min : float
        Minimum bin value.
    bin_max : float
        Maximum bin value.
    bin_num : int, optional
        Number of bins (default is 20).

    Returns
    -------
    tuple
        Probabilities, prior probabilities, and bin left edges.
    """
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
    """
    Computes the time spent near walls.

    Parameters
    ----------
    x_pos : numpy.ndarray
        X positions.
    y_pos : numpy.ndarray
        Y positions.
    wall_angs : list of float
        List of wall angles.
    dist_thresh : float, optional
        Distance threshold (default is 2).
    rad : float, optional
        Radius of the arena (default is 30.48).

    Returns
    -------
    tuple
        Percentages and in-zone flags.
    """
    percents = []
    in_zone = []
    for a in wall_angs:
        dtw = dist_to_wall(x_pos, y_pos, a)
        percents.append(100 * (sum(dtw < dist_thresh) / len(x_pos)))
        in_zone.append(dtw < dist_thresh)
    return percents, in_zone


def dist_to_wall(x, y, wall_deg, rad=30.48, num_pts=1000):
    """
    Computes the distance to a wall.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates.
    y : numpy.ndarray
        Y coordinates.
    wall_deg : float
        Wall angle in degrees.
    rad : float, optional
        Radius of the arena (default is 30.48).
    num_pts : int, optional
        Number of points to sample along the wall (default is 1000).

    Returns
    -------
    numpy.ndarray
        Array of distances to the wall.
    """
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
