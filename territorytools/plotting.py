import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from behavior import get_diadic_behavior
from urine import urine_across_time


def add_territory_circle(ax, block=None, rad=32):
    circ = Circle((0, 0), radius=rad, facecolor=(0.9, 0.9, 0.9), edgecolor=(0.2, 0.2, 0.2), linestyle='--')
    ax.add_patch(circ)
    if block == 'block0':
        ax.plot([0, 0], [0, -rad], color=(0.2, 0.2, 0.2), linestyle='--')
        ax.plot([0, rad*np.sin(np.radians(60))], [0, rad*np.cos(np.radians(60))], color=(0.2, 0.2, 0.2), linestyle='--')
        ax.plot([0, rad * np.sin(np.radians(-60))], [0, rad * np.cos(np.radians(-60))], color=(0.2, 0.2, 0.2),
                linestyle='--')


def plot_run(run_data, md, ax=None):
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
    def calc_cdf(data):
        return np.cumsum(data) / np.sum(data)

    for d in group_data:
        cols = ['tab:blue', 'tab:orange']
        for i in range(len(d)):
            ut = urine_across_time(d[i]['urine_data'], len_s=77050 / 40)
            ut[ut > 1000] = np.nan
            ut = (ut > 0).astype(int)
            cdf = calc_cdf(ut)
            t = np.linspace(0, len(ut) / 40, len(ut))
            ax_list[i].plot(t, cdf, c=cols[i])


def urine_prob_dep(run_data, run_info):
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