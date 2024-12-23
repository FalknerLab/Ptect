import os

import matplotlib.pyplot as plt
import territorytools.urine as ut
import territorytools.process as tdt
import territorytools.ttclasses as tdc
from territorytools.plotting import add_territory_circle
import numpy as np
import cv2
import h5py


casp_info = {'dac001': 'Caspase+',
             'dac002': 'Control',
             'dac003': 'Caspase+',
             'dac004': 'Caspase+',
             'dac005': 'Control',
             'dac006': 'Caspase+',
             'dac007': 'Caspase+',
             'dac008': 'Control',
             'dac009': 'Caspase+',
             'dac010': 'Control',
             'dac011': 'Caspase+',
             'dac012': 'Caspase+',
             'dac013': 'Control',
             'dac014': 'Control'}


def append_casp_ids(exp_obj):
    for r in exp_obj.runs:
        r.add_key_val('caspase', casp_info[r.get_key_val('resident')])


def make_casp_fig(exp_obj):
    pref_exp = exp_obj.compute_across_runs(tdt.compute_preferences)
    f, ax = plt.subplots()

    def scatter_casp(name, data, info):
        for p in data:
            if name == 'Caspase+':
                ax.scatter([0.25, 1.25, 2.25], p[0], color='r')
            else:
                ax.scatter([0, 1, 2], p[0], color='b')

    pref_exp.compute_across_groups('Caspase', scatter_casp)
    plt.ylim(0, 0.6)
    plt.show()


def process_pee(data_files):
    for d in data_files:
        pts = ut.sleap_to_fill_pts(d)
        avi = d.split('.')[:-1][0] + '.avi'
        out_tf = d.split('.')[:-1][0] + '_times.npy'
        out_ef = d.split('.')[:-1][0] + '_marks.npy'
        pt = ut.Peetector(avi, pts)
        pee_times, pee_evts = pt.peetect_frames()
        np.save(out_tf, pee_times)
        np.save(out_ef, pee_evts)


def make_pee_exp(pee_files):
    run_list = []
    for p in pee_files:
        file_parts = p.split('_')
        if file_parts[-1] == 'times.npy':
            times_data = np.load(p)
            xy_data = np.load('_'.join(file_parts[:-1]) + '_marks.npy', allow_pickle=True)
            urine_data = ut.expand_urine_data(xy_data[:, 0], times=times_data)
            run_data = '_'.join(file_parts[1:-2]).lower()
            run_info = tdc.make_dict(run_data, '_')
            # fill_mask = fill_walls(urine_data)
            # cor_xy = plot_rotated(fill_mask, run_info)
            run_obj = tdc.BasicRun(urine_data, run_info)
            run_list.append(run_obj)
    return tdc.BasicExp(run_list)

def plot_rotated(run_mask, run_info):
    mask_inds = np.fliplr(np.argwhere(run_mask))
    urine_cm = ut.urine_px_to_cm(mask_inds)
    orient = run_info['orientation'].lower()
    cor_x, cor_y = tdt.rotate_xy(urine_cm[:, 0], urine_cm[:, 1], orient)
    cor_xy = np.vstack((cor_x, cor_y)).T
    # f, axs = plt.subplots(1, 2)
    # ut.plot_urine_xys(urine_cm, ax=axs[0])
    # ut.plot_urine_xys(cor_xy, ax=axs[1])
    # axs[0].set_title(run_info)
    # plt.show()
    return cor_xy


def fill_walls(run_data):
    cnt = [330, 210]
    pnts = [[325, 440],
            [150, 140],
            [150, 120],
            [506, 97],
            [506, 115],
            [506, 120]]
    run_mask = np.zeros((480, 640)).astype(int)
    xys = ut.proj_urine_across_time(run_data, thresh=80).astype(int)
    run_mask[xys[:, 1], xys[:, 0]] = 1
    mask = run_mask.copy()
    xs = []
    ys = []
    for p in pnts:
        fill_x = np.linspace(p[0], cnt[0], 40)
        fill_y = np.linspace(p[1], cnt[1], 40)
        for px, py in zip(fill_x, fill_y):
            cv2.floodFill(mask, None, (int(px), int(py)), 0)
        xs.append(fill_x)
        ys.append(fill_y)
    # f, axs = plt.subplots(1, 2)
    # axs[0].imshow(run_mask)
    # axs[0].scatter(xs, ys)
    # axs[1].imshow(mask)
    # axs[0].set_title(run_info)
    # plt.show()
    return mask


def sum_urine_group(g_id, g_data, g_info, x):
    for g in g_data:
        this_area = sum(g[:, 0] < 0)
        plt.scatter()


def make_exp(o_files, p_files):
    run_list = []
    for p in p_files:
        p_l = p.lower()
        file_parts = p_l.split('_')
        if file_parts[-1] == 'marks.npy' and file_parts[8] == '0':
            run_data = '_'.join(file_parts[1:-2])
            run_data2 = '_'.join(file_parts[1:-6])
            for o in o_files:
                o_fp = o.split('_')
                orun_data = '_'.join(o_fp[1:-4])
                orun_data2 = '_'.join(o_fp[1:-8])
                if orun_data2 == run_data2:
                    run_info = tdc.make_dict(run_data, '_')
                    slp = h5py.File(o, 'r')
                    slp_pts = slp['tracks'][0]
                    o_x, o_y, _, _, _ = tdt.get_territory_data(slp_pts, rot_offset=run_info['orientation'])
                    u_xy = np.load(p, allow_pickle=True)
                    # fill_mask = fill_walls(u_xy)
                    cor_xy = plot_rotated(u_xy, run_info)
                    run_dict = {'uxy': cor_xy,
                                'x': o_x,
                                'y': o_y}
                    run_obj = tdc.BasicRun(run_dict, run_info)
                    run_list.append(run_obj)
    return tdc.BasicExp(run_list)


def plot_dtu(run_dict):
    xy = np.vstack((run_dict['x'], run_dict['y'])).T
    dtu = ut.dist_to_urine(xy, run_dict['uxy'])
    plt.figure()
    plt.scatter(xy[:, 0], xy[:, 1], c=dtu)
    plt.scatter(run_dict['uxy'][:, 0], run_dict['uxy'][:, 1], marker='+', c='r')
    plt.show()


def dtu_per_group(g_id, g_data, g_info, ax):
    acc = np.array([])
    for r in g_data:
        xy = np.vstack((r['x'], r['y'])).T
        dtu = ut.dist_to_urine(xy, r['uxy'])
        acc = np.hstack((acc, dtu))
    # hist = np.histogram(acc, bins=120, range=[0, 30])
    ax.hist(acc, bins=120, range=[0, 30], density=True, stacked=True)
    ax.set_title(g_id)


def sum_per_group(g_id, g_data, g_info, x):
    for r in g_data:
        print(g_id)
        sum_fam = np.shape(r)[0]
        plt.scatter(x, sum_fam)


def plot_each(gi, run_data, g_i, args):
    ax = args[1]
    add_territory_circle(ax)
    for r, i in zip(run_data, g_i):
        xys = ut.proj_urine_across_time(r, thresh=40)
        c_x, c_y = ut.xy_to_cm(xys)
        r_x, r_y = ut.rotate_xy(c_x, c_y, i['orientation'])
        ax.scatter(r_x, r_y, c=args[0])
        ax.set_title(i['caspase'])
        ax.set_xlim(-32, 32)
        ax.set_ylim(-32, 32)

def plot_b0(gi, run_data, g_i, args):
    ax = args[1]
    add_territory_circle(ax, block='block0')
    for r, i in zip(run_data, g_i):
        mask = fill_walls(r)
        xys = np.fliplr(np.argwhere(mask))
        c_x, c_y = ut.xy_to_cm(xys)
        r_x, r_y = ut.rotate_xy(c_x, c_y, i['orientation'])
        ax.scatter(r_x, r_y, c=args[0])
        ax.set_title(i['caspase'])
        ax.set_xlim(-32, 32)
        ax.set_ylim(-32, 32)


if __name__ == "__main__":

    #PEE DATA PLOTS

    # slp_data = get_sleap_files(path='data/thermal/slp', use_gui=True)
    # for s in slp_data:
    #     su.slp_to_h5(s)
    root_dir = 'D:\\TerritoryTools\\data\\casp\\thermal\\output'
    pee_files = os.listdir(root_dir)
    new_files = [os.path.join(root_dir, p) for p in pee_files]
    # process_pee(pee_files)

    # op_files = get_sleap_files(path='data/h5', use_gui=False)
    # casp_exp = make_exp(op_files, pee_files)
    # append_casp_ids(casp_exp)
    # plt.figure()
    # casp_exp.compute_across_groups('caspase', sum_per_group, [1, 2], arg_per_group=True)
    # plt.show()
    # f, axs = plt.subplots(2, 1)
    # casp_exp.compute_across_groups('caspase', dtu_per_group, axs, arg_per_group=True)
    # plt.show()
    # casp_exp.compute_across_runs(plot_dtu)
    pee_exp = make_pee_exp(new_files)
    append_casp_ids(pee_exp)
    pee_exp = pee_exp.filter_by_group('block', '0')
    f, axs = plt.subplots(1, 2)
    pee_exp.compute_across_groups('caspase', plot_b0, [('tab:green', axs[0]), ('grey', axs[1])], arg_per_group=True)
    # pee_exp.compute_across_groups('caspase', sum_per_group, [1, 2], arg_per_group=True)
    plt.show()


    # mask_exp = pee_exp.compute_across_runs(fill_walls, pass_info=True)
    # xy_exp = mask_exp.compute_across_runs(plot_rotated, pass_info=True)
    # f, axs = plt.subplots(1, 2)
    # xy_exp.compute_across_groups('Caspase', ut.plot_all_urine_xys, [('r', axs[0]), ('b', axs[1])], arg_per_group=True)

    #OPTICAL DATA (XY) PLOTS

    # bias_files = get_sleap_files(path='data/h5', use_gui=False)
    # exp = import_data_as_experiment(bias_files)
    # append_casp_ids(exp)
    # make_casp_fig(exp)

    # def get_vals(x):
    #     return x[0]

    # fig, axs = plt.subplots(1, 2)
    # exp.compute_across_groups('Caspase', tdt.group_heatmap, axs, arg_per_group=True)


    # bias_exp = exp.compute_across_runs(tdt.compute_preferences)
    # bias_exp = bias_exp.compute_across_runs(get_vals)
    # cont = np.array(bias_exp.filter_by_group('Caspase', 'Control').get_run_data())
    # casp = np.array(bias_exp.filter_by_group('Caspase', 'Caspase+').get_run_data())
    # ttest_res = ttest_ind(cont[:, 2], casp[:, 2])
    # print(ttest_res)
