from matplotlib.gridspec import GridSpec
from scipy.stats import alpha

from territorytools.process import make_run_data_struct, find_file_recur
from territorytools.plotting import add_territory_circle, territory_heatmap
from territorytools.ttclasses import BasicExp, BasicRun
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_all_data():

    # Make main figure
    f = plt.figure()

    # Prepare GridSpec for subplots
    gs = plt.GridSpec(5, 7)

    b0_axs = [gs[:2, 0], gs[:2, 1], gs[:2, 2], gs[2, :3], gs[3, :3]]
    ax_list = []
    for b0a in b0_axs:
        a = f.add_subplot(b0a)
        ax_list.append(a)

    add_block_plots(ax_list)
    plt.show()


def add_block_plots(ax_list, block='block0'):
    xy_ax = ax_list[0]
    u_ax = ax_list[1]
    for a in (xy_ax, u_ax):
        add_territory_circle(a, block=block)


def get_dataset_as_exp(dataset_dir):
    files = os.listdir(dataset_dir)
    runs = []
    for file in files:
        this_path = os.path.join(dataset_dir, file)
        run_data, metadata = make_run_data_struct(this_path, kpms_csv='D:/ptect_dataset/kpms/ptect4_kpms.csv')
        if metadata is not None:
            r = BasicRun(run_data, metadata)
            runs.append(r)
    return BasicExp(runs)


def plot_all_block1(g_id, g_d, g_info):
    s = int(np.ceil(np.sqrt(len(g_d))))
    f, axs = plt.subplots(s, s)
    axs = axs.ravel()
    for ind, d in enumerate(g_d):
        if len(d['Exploration']) > 0:
            x = d['Exploration'][0]['x_cm']
            y = d['Exploration'][0]['y_cm']
            territory_heatmap(x, y, ax=axs[ind], vmax=0.01)
    plt.show()


def plot_kpms(g_id, g_d, g_info):
    good_s = [0, 2, 13, 20, 30, 33, 34, 37, 40, 42, 47, 48, 52, 57, 58, 59, 61, 66, 68, 75, 77, 79, 80, 82, 86, 88, 91]
    syl_accs = [[] for n in range(len(good_s))]
    for d in g_d:
        if len(d['Exploration']) > 0:
            # f, axs = plt.subplots(3, len(good_s))
            # axs = axs.ravel()
            x = d['Exploration'][0]['x_cm']
            y = d['Exploration'][0]['y_cm']
            syls = d['Exploration'][0]['kpms_syllables']
            for i, s in enumerate(good_s):
                only_1 = syls == s
                syl_count = territory_heatmap(x[only_1], y[only_1], vmin=0, vmax=0.05, density=False)
                tot_time_s = territory_heatmap(x, y, vmin=0, vmax=0.05, density=False) / 40
                # axs[2, i].imshow(syl_dens/tot_time, extent=(-32, 32, -32, 32), interpolation='none', origin='lower', cmap='plasma')
                # axs[0, i].set_title(f'Syllable {s}')
                syl_accs[i].append(syl_count/tot_time_s)
    f, axs = plt.subplots(5, 6)
    axs = axs.ravel()
    ax = 0
    mean_freq = None
    for ax, acc in enumerate(syl_accs):
        mean_freq = np.mean(acc, axis=0)
        im_obj =axs[ax].imshow(mean_freq, extent=(-32, 32, -32, 32), interpolation='none', origin='lower', cmap='plasma')
        plt.colorbar(im_obj, ax=axs[ax], label='syls / s')
        axs[ax].set_title(f'Syllable {good_s[ax]}')
    # im_obj = axs[ax+1].imshow(mean_freq, extent=(-32, 32, -32, 32), interpolation='none', origin='lower', cmap='plasma', vmin=0,
    #                vmax=2)

    plt.show()


def plot_kpms_ind(g_id, g_d, g_info):
    good_s = [0, 2, 13, 20, 30, 33, 34, 37, 40, 42, 47, 48, 52, 57, 58, 59, 61, 66, 68, 75, 77, 79, 80, 82, 86, 88, 91]
    gs = GridSpec(4, 15 + 2)
    for d, info in zip(g_d, g_info):
        if len(d['Exploration']) > 0:
            x = d['Exploration'][0]['x_cm']
            y = d['Exploration'][0]['y_cm']
            syls = d['Exploration'][0]['kpms_syllables']
            b0_fold = '\\'.join(os.path.split(d['Exploration'][0]['data_folder'])[:-1]) + '\\Block0'
            p_b0 = find_file_recur(b0_fold, 'ptdata.npy')
            p_b1 = find_file_recur(d['Exploration'][0]['data_folder'], 'ptdata.npy')
            p_data_b0 = np.load(p_b0)
            p_data_b1 = np.load(p_b1)

            f = plt.figure(figsize=(24, 12))
            ax_occ = f.add_subplot(gs[:2, :2])
            tot_time = territory_heatmap(x, y, ax=ax_occ, vmin=0, vmax=0.05)

            ax_p = f.add_subplot(gs[2:, :2])
            territory_heatmap([np.nan, np.nan], [np.nan, np.nan], ax=ax_p)
            self_marks = p_data_b0[0, :] < 0
            sm = ax_p.scatter(p_data_b0[0, self_marks], p_data_b0[1, self_marks], c='tab:blue', s=1, alpha=0.25)
            om = ax_p.scatter(p_data_b0[0, ~self_marks], p_data_b0[1, ~self_marks], c='tab:orange', s=1, alpha=0.25)
            b1m = ax_p.scatter(p_data_b1[0, :], p_data_b1[1, :], c='tab:green', s=1, alpha=0.25)
            plt.figlegend(handles=[sm, om, b1m], labels=['self', 'other', 'exploration'], title='Marks', loc='lower left')

            for i, s in enumerate(good_s):
                if i < 15:
                    syl_ax = f.add_subplot(gs[0, i+2])
                    norm_ax = f.add_subplot(gs[1, i + 2])
                else:
                    syl_ax = f.add_subplot(gs[2, (i-15)+2])
                    norm_ax = f.add_subplot(gs[3, (i-15) + 2])
                only_1 = syls == s
                syl_dens = territory_heatmap(x[only_1], y[only_1], ax=syl_ax, vmin=0, vmax=0.05)
                norm_ax.imshow(syl_dens/tot_time, extent=(-32, 32, -32, 32), interpolation='none', origin='lower', cmap='plasma')
                syl_ax.set_title(f'Syllable {s}')
            # plt.tight_layout()
            out_file = os.path.split(p_b1)
            sub = info['Territory']['self']
            out_file = f'{sub}_syllables.png'
            print(out_file)
            plt.subplots_adjust(top=0.989, bottom=0.011, left=0.021, right=0.994, hspace=0.048, wspace=0.565)
            plt.savefig(out_file)
            # plt.show()


if __name__ == '__main__':
    dataset_dir = os.path.abspath("D:/ptect_dataset/")
    dataset = get_dataset_as_exp(dataset_dir)
    dataset.compute_across_groups('Territory/experiment_id', plot_kpms_ind)

