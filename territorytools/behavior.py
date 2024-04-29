import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb


def get_territory_data(sleap_pts, rot_offset=0, px_per_ft=350, ref_point=(638, 504)):
    head_angles = get_head_direction(sleap_pts)
    head_angles = []
    mouse_cent = np.nanmean(sleap_pts, axis=1).T
    x, y = xy_to_cm(mouse_cent, center_pt=ref_point, px_per_ft=px_per_ft)
    cent_x, cent_y = rotate_xy(x, y, rot_offset)
    rel_xy = np.vstack((x, y)).T
    dist_vec = np.linalg.norm(rel_xy[1:, :] - rel_xy[:-1, :], axis=1) / (px_per_ft/30.48)
    dist_vec = np.hstack(([0], dist_vec))
    return cent_x, cent_y, head_angles, dist_vec, sleap_pts


def rotate_xy(x, y, rot):
    rot_dict = {'rni': 0,
                'none': 0,
                'irn': 120,
                'nir': 240}
    rot_deg = 0
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


def get_head_direction(mouse_data, in_deg=False):
    md_copy = np.copy(mouse_data)
    md_copy[1, :, :] = -md_copy[1, :, :]
    xy_nose = md_copy[:, 0, :] - md_copy[:, 1, :]
    angs = np.arctan2(xy_nose[1, :], xy_nose[0, :])
    if in_deg:
        angs = np.degrees(angs)
    return angs


def compute_preferences(exp_data, walls=None):
    if walls is None:
        walls = [-np.pi / 2, 5 * np.pi / 6, np.pi / 6]
    x = exp_data[0]
    y = exp_data[1]
    t = np.arctan2(y, x)
    ter_a = np.logical_or(t < walls[0], t > walls[1])
    ter_b = np.logical_and(t > walls[0], t < walls[2])
    ter_c = np.logical_and(t > walls[2], t < walls[1])
    prefs = np.zeros(len(walls))
    for i, ter in enumerate((ter_a, ter_b, ter_c)):
        prefs[i] = np.sum(ter) / len(t)
    return prefs, ter_a, ter_b, ter_c


def interp_behavs(*args):
    out_behavs = []
    for b in args:
        interp_b = b.copy()
        nans = np.isnan(interp_b)
        interp_vals = np.interp(nans.nonzero()[0], (~nans).nonzero()[0], interp_b[~nans])
        interp_b[nans] = interp_vals
        out_behavs.append(interp_b)
    return out_behavs


def plot_prefs(g, list_pref, info, ax):
    for p_data, this_i in zip(list_pref, info):
        p = p_data[0]
        x = np.arange(3)
        ax.plot(x, p, label=this_i['Resident'])
    plt.legend(bbox_to_anchor=(1.05, 1.0))
    ax.set_ylim(0,0.75)
    ax.set_title(g)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Resident', 'Intruder', 'Neutral'])


def group_heatmap(g, group_data, group_info, ax):
    x_acc = np.array([])
    y_acc = np.array([])
    for run_data in group_data:
        x, y = run_data[:2]
        x_acc = np.hstack((x_acc, x))
        y_acc = np.hstack((y_acc, y))
    h = ax.hist2d(x_acc, y_acc, bins=25,
               range=[[np.nanmin(x_acc), np.nanmax(x_acc)], [np.nanmin(y_acc), np.nanmax(y_acc)]],
               density=True, vmin=0, vmax=0.001, cmap='gnuplot')
    plt.colorbar(h[3], ax=ax)
    ax.set_title(g)


def plot_prefs_across_group(g, group_data, group_info, ax, ids):
    pref_mat = np.array(group_data)
    marker_list = list(Line2D.markers.keys())
    marker_ind = np.argwhere(np.equal(ids, g))[0][0]
    mark = marker_list[marker_ind]
    ax.plot(pref_mat[:, 0], marker=mark, color='g')
    ax.plot(pref_mat[:, 1], marker=mark, color=[0.7, 0.3, 0])
    ax.plot(pref_mat[:, 2], marker=mark, color=[0.3, 0.3, 0.3])


def generate_color(subject_id):
    print(hash(subject_id))


def add_territory_circle(ax, walls=None, is_polar=False, arena_width_cm=28):
    if walls is None:
        walls = [-np.pi / 2, 5 * np.pi / 6, np.pi / 6]


def plot_bias(g, prefs, info):
    thetas = np.array([7 * np.pi / 6, 11 * np.pi / 6, np.pi / 2])
    xs = []
    ys = []
    for p in prefs:
        r = p[0]
        x = r @ np.cos(thetas)
        y = r @ np.sin(thetas)
        xs.append(x)
        ys.append(y)
    plt.plot(xs, ys, 'x-b')


def avg_angs(exp_data):
    #Get average angle of head direction data
    avg_s = np.nanmean(np.sin(exp_data[2]))
    avg_c = np.nanmean(np.cos(exp_data[2]))
    return np.arctan2(avg_s, avg_c)


def xy_func(x):
    return np.vstack(([x[0]], [x[1]])).T


def get_t(x, r_i):
    t = np.arctan2(x[1], x[0])
    return t


def plot_territory(ax, t, r):
    walls = [-np.pi / 2, 5 * np.pi / 6, np.pi / 6]
    num_f = len(t)
    ter_a = np.logical_or(t < walls[0], t > walls[1])
    ter_b = np.logical_and(t > walls[0], t < walls[2])
    ter_c = np.logical_and(t > walls[2], t < walls[1])
    cmap = np.tile([0.6, 0.6, 0.6], (num_f, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    cmap[ter_b, :] = [0, 0.6, 0]
    cmap[ter_a, :] = [0.7, 0.6, 0.3]
    # ax.scatter(t, r, c=cmap, s=3)
    h, te, re, _ = ax.hist2d(t, r, bins=50, range=[[np.nanmin(t), np.nanmax(t)], [np.nanmin(r), np.nanmax(r)]],
                             density=True)
    return h, te, re


def compute_over_spatial_bin(xpos, ypos, data, func, bins=20, range=None):
    if range is None:
        range = [[-400, 400], [-400, 400]]
    _, xedges, yedges = np.histogram2d(xpos, ypos, bins=bins, range=range)
    out_hist = np.zeros((len(xedges) - 1, len(yedges) - 1))
    for i, xe in enumerate(xedges[:-1]):
        in_x = np.logical_and(xpos > xe, xpos < xedges[i + 1])
        for j, ye in enumerate(yedges[:-1]):
            in_y = np.logical_and(ypos > ye, ypos < yedges[j + 1])
            in_bin = np.logical_and(in_x, in_y)
            out_hist[i, j] = func(data[in_bin])
    return out_hist, xedges, yedges


def get_chance_line(group_ts):
    np.random.seed(5)
    group_ts = np.degrees(group_ts)
    n = np.shape(group_ts)[0]
    ang_options = np.arange(-170, 170, 10)
    hist_acc = []
    num_i = 100
    for i in range(num_i):
        ang_inds = np.random.randint(0, len(ang_options), n)
        these_angs = ang_options[ang_inds]
        shift_ts = group_ts + np.expand_dims(these_angs, axis=1)
        fix_inds = np.where(shift_ts > 180)
        shift_ts[fix_inds] = shift_ts[fix_inds] - 360
        fix_inds1 = np.where(shift_ts < -180)
        shift_ts[fix_inds1] = shift_ts[fix_inds1] + 360
        shift_ts = np.radians(shift_ts)
        for j in range(n):
            vals, bin_edges = np.histogram(shift_ts[j, :], bins=36, range=[-np.pi, np.pi])
            norm_vals = vals/sum(vals)
            hist_acc.append(norm_vals)
    out_acc = np.array(hist_acc)
    chance_line = 100*np.mean(out_acc, axis=0)
    return chance_line


def hist_t(g, ts, info):
    group_t = np.nan*np.zeros((len(ts), len(ts[0])))
    count_acc = []
    for ind, t in enumerate(ts):
        group_t[ind, :] = t
        vals, bin_edges = np.histogram(group_t, bins=36, range=[-np.pi, np.pi])
        norm_vals = 100*(vals/sum(vals))
        count_acc.append(norm_vals)

    count_acc = np.array(count_acc)
    mean_vals = np.mean(count_acc, axis=0)
    dev = np.std(count_acc, axis=0)
    n = np.shape(count_acc)[0]
    sem = 1.96*(dev/np.sqrt(n))
    ax = plt.subplot(projection='polar')
    r_edge = 1.2*max(mean_vals)
    pts = [-np.pi/2, 5*np.pi/6, np.pi/6]
    for p in pts:
        ax.plot([0, p], [0, r_edge], ':k', linewidth=2)

    # cl = get_chance_line(group_t)
    cl = 100*np.ones(len(mean_vals))/len(mean_vals)
    vals = np.hstack((mean_vals, mean_vals[0]))
    sem = np.hstack((sem, sem[0]))
    cl = np.hstack((cl, cl[0]))
    ax.plot(bin_edges, cl, color=[0, 0, 0], linewidth=2)
    ax.plot(np.linspace(np.pi/6, 5*np.pi/6, 120), r_edge*np.ones(120), color=[0.5, 0.5, 0.5], linewidth=2)
    ax.plot(np.linspace(5*np.pi/6, 1.5*np.pi, 120), r_edge*np.ones(120), color=[0.2, 0.6, 0.2], linewidth=2)
    ax.plot(np.linspace(np.pi/6, -np.pi/2, 120), r_edge*np.ones(120), color=[0.8, 0.5, 0.2], linewidth=2)
    ax.plot(bin_edges, vals + sem, color=[0.4, 0.4, 0.8], linewidth=2, linestyle='--')
    ax.plot(bin_edges, vals - sem, color=[0.4, 0.4, 0.8], linewidth=2, linestyle='--')
    ax.plot(bin_edges, vals, color=[0, 0, 0.8], linewidth=4)
    ax.set_rlabel_position(0)
    ax.set_xticklabels([])
    ax.set_yticks(ax.get_yticks()[:-1])
    ax.grid(axis="x")
    plt.show()


def behavioral_raster(rasters, ax=plt.gca, fs=1):
    cols = ['tab:green', 'tab:orange', 'tab:gray', 'c', 'm', 'r', 'k']
    len_behav, num_rasts = np.shape(rasters)
    rast_im = 255*np.ones((1, len_behav, 3))
    for i in range(num_rasts):
        col = to_rgb(cols[i])
        inds = np.where(rasters[:, i])[0]
        rast_im[:, inds, :] = col
    ax.imshow(rast_im, aspect='auto')
    ax.set_xlim([0, len_behav])
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    x_ticks = ax.get_xticks()
    x_ticks = x_ticks[x_ticks < len_behav]
    ax.set_xticks(x_ticks, labels=x_ticks/fs)
    ax.get_yaxis().set_visible(False)


def make_one_hot(*args):
    t = len(args[0])
    num_features = len(args)
    one_hot = np.zeros((t, num_features))
    for f in range(num_features):
        one_hot[:, f] = args[f]
    return one_hot
