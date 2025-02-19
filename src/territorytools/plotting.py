import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from behavior import get_diadic_behavior, compute_over_spatial_bin, avg_angs
from urine import urine_across_time


def add_territory_circle(ax, block=None, rad=30.48, facecolor=None):
    """
    Adds a territory circle to the given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to add the circle to.
    block : str, optional
        Block type to add specific lines (default is None).
    rad : float, optional
        Radius of the circle in centimeters (default is 30.48).
    facecolor : tuple, optional
        Face color of the circle (default is None).

    Returns
    -------
    None
    """
    if facecolor is None:
        facecolor = (0.9, 0.9, 0.9)
    circ = Circle((0, 0), radius=rad, facecolor=facecolor, edgecolor=(0.2, 0.2, 0.2), linestyle='--')
    ax.add_patch(circ)
    if block == 'block0':
        c = (0, 0, 0)
        ax.plot([0, 0], [0, -rad], color=c, linestyle='--')
        ax.plot([0, rad*np.sin(np.radians(60))], [0, rad*np.cos(np.radians(60))], color=c, linestyle='--')
        ax.plot([0, rad * np.sin(np.radians(-60))], [0, rad * np.cos(np.radians(-60))], color=c,
                linestyle='--')
    ax.set_xlim(-rad*1.1, rad*1.1)
    ax.set_ylim(-rad * 1.1, rad * 1.1)


def plot_run(run_data, md, ax=None):
    """
    Plots the run data on the given axis.

    Parameters
    ----------
    run_data : list of dict
        List of dictionaries containing run data for each mouse.
    md : dict
        Metadata associated with the run.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on (default is None).

    Returns
    -------
    None
    """
    if ax is None:
        ax = plt.gca()
    cols = ['tab:blue', 'tab:orange']
    dist_btw_mice, looking_ang = [], []
    if md['block'] == '0':
        dist_btw_mice, looking_ang = get_diadic_behavior(run_data[0]['x_cm'], run_data[0]['y_cm'], run_data[0]['angle'],
                                                         run_data[1]['x_cm'], run_data[1]['y_cm'])
    ind = 0
    for m, d in enumerate(run_data):
        ut = urine_across_time(d['urine_data'], len_s=len(d['x_cm'])/40)
        ut[ut > 1000] = np.nan
        ut = (ut > 0).astype(int)
        b_feats = [d['x_cm'], d['y_cm'], d['velocity'], d['angle'], ut]
        for ft in b_feats:
            norm_f = (ft - np.nanmin(ft)) / (np.nanmax(ft) - np.nanmin(ft))
            ax.plot(norm_f - ind, c=cols[m])
            ind += 1.2

    if len(dist_btw_mice) > 0:
        fts = [dist_btw_mice, looking_ang]
        for f in fts:
            norm_f = (f - np.nanmin(f)) / (np.nanmax(f) - np.nanmin(f))
            ax.plot(norm_f - ind, c='k')
            ind += 1.2


def plot_cdfs(group_id, group_data, group_info, ax_list):
    """
    Plots the cumulative distribution functions (CDFs) for the given group data.

    Parameters
    ----------
    group_id : any
        Identifier for the group.
    group_data : list of list of dict
        List of lists containing data for each group.
    group_info : list of dict
        List of dictionaries containing metadata for each group.
    ax_list : list of matplotlib.axes.Axes
        List of axes to plot on.

    Returns
    -------
    None
    """
    def calc_cdf(data):
        return np.cumsum(data) / np.sum(data)

    for d in group_data:
        cols = ['tab:blue', 'tab:orange']
        for i in range(len(d)):
            ut = urine_across_time(d[i]['urine_data'], len_s=len(d[i]['x_cm'])/40)
            u_on = (np.diff(ut) > 0.5).astype(int)
            cdf = calc_cdf(u_on)
            t = np.linspace(0, len(u_on) / 40, len(u_on))
            ax_list[i].plot(t, cdf, c=cols[i])


def urine_prob_dep(run_data, run_info):
    """
    Plots the probability of urination depending on the distance between mice.

    Parameters
    ----------
    run_data : list of dict
        List of dictionaries containing run data for each mouse.
    run_info : dict
        Metadata associated with the run.

    Returns
    -------
    None
    """
    x_fix = [1, -1]
    o_ind = [1, 0]
    f, axs = plt.subplots(2, 1)
    for i, (d, x, o) in enumerate(zip(run_data, x_fix, o_ind)):
        ut = urine_across_time(d['urine_data'], len_s=77050 / 40)
        ut = (ut > 0).astype(int)
        o_x = x*run_data[o]['x_cm']
        ut = ut[:len(o_x)]
        tot_u = np.sum(ut)
        dist_between, _ = get_diadic_behavior(run_data[0]['x_cm'], run_data[0]['y_cm'], run_data[0]['angle'],
                                              run_data[1]['x_cm'], run_data[1]['y_cm'])
        bs = np.linspace(0, 60, 61)
        bin_ox = np.digitize(dist_between, bins=bs)
        # bs = np.linspace(0, 32, 33)
        # bin_ox = np.digitize(o_x, bins=bs)
        probs = []
        prior = []
        for b in bs:
            this_bin = bin_ox == b
            probs.append(np.sum(ut[this_bin]) / tot_u)
            prior.append(np.sum(this_bin)/len(bin_ox))
        prior = 100*np.array(prior)
        probs = 100*np.array(probs)
        axs[i].bar(bs, probs, align='edge', color='b', alpha=0.5)
        axs[i].bar(bs, prior, align='edge', color='k', alpha=0.5)
        axs[i].plot(bs, probs / prior)
    title = str(run_info)
    delimiters = ["\'", "{", "}", ":", ',']

    for delimiter in delimiters:
        title = " ".join(title.split(delimiter))
    axs[0].set_title(title)


def show_urine_segmented(expand_urine, labels):
    """
    Shows a 3D scatter plot of segmented urine data.

    Parameters
    ----------
    expand_urine : numpy.ndarray
        Array of urine data points.
    labels : numpy.ndarray
        Array of labels for the urine data points.

    Returns
    -------
    None
    """
    e = [20, 0]
    a = [-45, 0]
    for i in range(2):
        f = plt.figure()
        ax = f.add_subplot(projection='3d')
        ax.set_facecolor("white")
        arc_x = 30.48*np.sin(np.linspace(np.pi, 1*np.pi/3, 100))
        arc_y = 30.48 * np.cos(np.linspace(np.pi, 1*np.pi /3, 100))
        plt.plot(np.zeros(100), arc_x, arc_y, 'k--')
        plt.plot([0, 0], [0, 0], [0, -30.48], 'k--')
        plt.plot([0, 0], [0, 30.48*np.sin(np.pi/3)], [0, 30.48*np.cos(np.pi/3)], 'k--')
        ax.scatter(expand_urine[:, 0], expand_urine[:, 1], expand_urine[:, 2], c=labels, cmap='tab20')
        ax.set_ylim(-5, 32)
        ax.set_zlim(-32, 15)
        ax.view_init(elev=e[i], azim=a[i])


def show_urine_segmented_video(expand_urine, labels, window=300):
    """
    Shows a 3D scatter plot animation of segmented urine data over time.

    Parameters
    ----------
    expand_urine : numpy.ndarray
        Array of urine data points.
    labels : numpy.ndarray
        Array of labels for the urine data points.
    window : int, optional
        Window size for the animation (default is 300).

    Returns
    -------
    None
    """
    num_f = int(max(expand_urine[:, 0])) + window
    f = plt.figure()
    ax = f.add_subplot(projection='3d')
    ax.set_xlim(0, window)
    ax.set_ylim(-32, 32)
    ax.set_zlim(-32, 32)
    s = ax.scatter([], [], [])
    set3 = cm.get_cmap('Set3', len(np.unique(labels))).colors

    def update(frame):
        data_in_win = np.logical_and(expand_urine[:, 0] > frame, expand_urine[:, 0] < frame + window)
        frame_data = expand_urine[data_in_win, :]
        if len(frame_data) > 0:
            s._offsets3d = (frame_data[:, 0] - frame, frame_data[:, 1], frame_data[:, 2])
            s.set_color(set3[labels[data_in_win], :])
        return s

    print('Running animation...')
    anim = FuncAnimation(fig=f, func=update, frames=num_f, interval=1)
    plt.show()
    # anim.save('urine3d.mp4', writer='ffmpeg', progress_callback=lambda i, n: print(f'Saving frame {i}/{n}'), fps=30)


def circle_hist(x, y, bins=36, ax=None, **kwargs):
    """
    Plots a circular histogram of the given data.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates.
    y : numpy.ndarray
        Y coordinates.
    bins : int, optional
        Number of bins for the histogram (default is 36).
    ax : matplotlib.axes.Axes, optional
        Axis to plot on (default is None).

    Returns
    -------
    None
    """
    if ax is None:
        ax = plt.subplot(projection='polar')

    theta = np.arctan2(y, x)
    vals, bin_edges = np.histogram(theta, bins=bins, range=[-np.pi, np.pi])
    norm_vals = 100*(vals/sum(vals))

    r_edge = 1.2*max(norm_vals)
    pts = [-np.pi/2, 5*np.pi/6, np.pi/6]
    for p in pts:
        ax.plot([0, p], [0, r_edge], ':k', linewidth=2)

    norm_vals = np.hstack((norm_vals, norm_vals[0]))
    # ax.plot(np.linspace(np.pi/6, 5*np.pi/6, 120), r_edge*np.ones(120), color=[0.5, 0.5, 0.5], linewidth=2)
    # ax.plot(np.linspace(5*np.pi/6, 1.5*np.pi, 120), r_edge*np.ones(120), color=[0.2, 0.6, 0.2], linewidth=2)
    # ax.plot(np.linspace(np.pi/6, -np.pi/2, 120), r_edge*np.ones(120), color=[0.8, 0.5, 0.2], linewidth=2)
    ax.plot(bin_edges, norm_vals, **kwargs)
    ax.set_rlabel_position(0)
    ax.set_xticklabels([])
    ax.set_yticks(ax.get_yticks()[:-1])
    ax.grid(axis="x")


def motion_flow_field(x, y, ax=None, future_frames=40, bin_s=50):
    """
    Plots a motion flow field of the given data.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates.
    y : numpy.ndarray
        Y coordinates.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on (default is None).
    future_frames : int, optional
        Number of future frames to consider (default is 40).
    bin_s : int, optional
        Bin size for the histogram (default is 50).

    Returns
    -------
    None
    """
    if ax is None:
        f, ax = plt.subplots(1, 1)

    xy_data = np.vstack((x, y)).T
    angs = []
    rads = []
    for cent0, cent1 in zip(xy_data[:-future_frames, :], xy_data[future_frames:, :]):
        rel_xy = cent1 - cent0
        heading = np.arctan2(rel_xy[1], rel_xy[0])
        rads.append(np.linalg.norm(rel_xy))
        angs.append(heading)
    hist, x_edges, y_edges = compute_over_spatial_bin(xy_data[:-future_frames, 0], xy_data[:-future_frames, 1],
                                        np.array(angs), avg_angs, range=[[-32, 32], [-32, 32]], bins=bin_s)
    hist_r, x_edges, y_edges = compute_over_spatial_bin(xy_data[:-future_frames, 0], xy_data[:-future_frames, 1],
                                        np.array(rads), np.nanmedian, range=[[-32, 32], [-32, 32]], bins=bin_s)
    ax.quiver(x_edges[:-1], y_edges[:-1], hist_r*np.sin(hist), hist_r*np.cos(hist))
    # ax.quiver(x_edges[:-1], y_edges[:-1], np.sin(hist), np.cos(hist))


def plot_mean_sem(data_mat: np.ndarray, ax=None, col='r', x=None):
    """
    Plots the mean and standard error of the mean (SEM) of the given data.

    Parameters
    ----------
    data_mat : numpy.ndarray
        Data matrix.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on (default is None).
    col : str, optional
        Color of the plot (default is 'r').
    x : numpy.ndarray, optional
        X coordinates (default is None).

    Returns
    -------
    None
    """
    if ax is None:
        ax = plt.gca()

    if x is None:
        x = np.linspace(0, data_mat.shape[0], data_mat.shape[0])

    mean = np.nanmean(data_mat, axis=1)
    dev = np.std(data_mat, axis=1)
    n = np.shape(data_mat)[1]
    sem = dev / np.sqrt(n)
    # sem_path0 = np.vstack((x, mean + sem)).T
    # sem_path1 = np.flipud(np.vstack((x, mean - sem)).T)
    # sem_path = np.vstack((sem_path0, sem_path1))
    # sem_patch = plt.Polygon(sem_path, facecolor=col, alpha=0.5)
    # ax.add_patch(sem_patch)
    ax.plot(x, mean, c=col)
    ax.plot(x, mean + sem, c=col, linestyle='--')
    ax.plot(x, mean - sem, c=col, linestyle='--')


def territory_heatmap(x, y, ax=None, bins=16, vmin=None, vmax=None, colorbar=False, limits=((-32, 32), (-32, 32)), density=True, aspect='auto'):
    """
    Plots a heatmap of the given territory data.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates.
    y : numpy.ndarray
        Y coordinates.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on (default is None).
    bins : int, optional
        Number of bins for the histogram (default is 16).
    vmin : float, optional
        Minimum value for the color scale (default is 0).
    vmax : float, optional
        Maximum value for the color scale (default is 200).
    colorbar : bool, optional
        Whether to show a colorbar (default is False).
    limits : tuple of tuple, optional
        Limits for the histogram (default is ((-32, 32), (-32, 32))).
    density : bool, optional
        Whether to normalize the histogram (default is True).
    aspect : str, optional
        Aspect ratio for the plot (default is 'auto').

    Returns
    -------
    numpy.ndarray
        Heatmap data.
    """
    im = np.histogram2d(x, y, bins=bins, range=limits)[0].T

    if density:
        im = im / len(x)
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = np.max(im)

    if ax is not None:
        im_obj = ax.imshow(im, extent=(-32, 32, -32, 32), interpolation='none', origin='lower', vmin=vmin, vmax=vmax, cmap='plasma', aspect=aspect)
        if colorbar:
            plt.colorbar(im_obj)
        ax.set_xlim(-32, 32)
        ax.set_ylim(-32, 32)
    return im


def draw_pvals(ax, pvals, x_pts):
    """
    Draws p-values on the given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    pvals : list of float
        List of p-values.
    x_pts : list of tuple
        List of x-coordinate pairs for the p-values.

    Returns
    -------
    None
    """
    start_y = ax.get_ylim()[1] * 1.01
    for p, x in zip(pvals, x_pts):
        if p < 0.05:
            ax.plot([x[0], x[1]], [start_y, start_y], c='k')
        sig_sym = ''
        if 0.01 <= p < 0.05:
            sig_sym = '*'
        if 0.001 <= p < 0.01:
            sig_sym = '**'
        if p < 0.001:
            sig_sym = '***'
        ax.text((x[1] - x[0])/2 + x[0], start_y * 1.01, sig_sym, fontweight='bold', horizontalalignment='center')
