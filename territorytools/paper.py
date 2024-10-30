import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.signal import correlate, correlation_lags
from process import valid_dir, import_all_data, find_territory_files
from ttclasses import BasicExp, BasicRun
from modeling import train_test_model, chunk_shuffle
from behavior import make_design_matrix, prob_over_x, rotate_xy
from scrap.urine_old import dist_to_urine, proj_urine_across_time
from plotting import add_territory_circle


def post_clean_urine(run_data: list[dict], thresh=100):
    for r in run_data:
        this_u = r['urine_data']
        unique_ts, urine_cnts = np.unique(this_u[:, 0], return_counts=True)
        bad_ts = unique_ts[urine_cnts > thresh]
        uts = this_u[:, 0]
        good_args = ~np.isin(uts, bad_ts)
        r['urine_data'] = this_u[good_args, :]
    return run_data


def load_dataset(data_dir):
    runs = []
    for f in os.listdir(data_dir):
        this_path = os.path.join(data_dir, f)
        if valid_dir(this_path):
            print(this_path)
            run_data, run_info = import_all_data(this_path, show_all=True)
            try:
                make_design_matrix(run_data, run_info, norm='0-1')
                this_run = BasicRun(run_data, run_info)
                runs.append(this_run)
            except IndexError:
                print(this_path)
    return BasicExp(runs)


def make_full_design_matrix(exp_data, norm='0-1'):
    all_data = []
    for r in exp_data.runs:
        try:
            design = make_design_matrix(r.data, r.info, norm=norm)
            all_data.append(design)
        except IndexError:
            z = 0
    all_data = np.vstack(all_data)
    return all_data


def urine_prob_group(exp_data):
    form_data = exp_data.filter_by_group('Territory/block', '0')
    form_design = make_full_design_matrix(form_data, norm=None)
    urine_inds = [4, 9, 9, 4, 4, 9]
    x_inds = [0, 5, 0, 5, 10, 10]
    _, axs = plt.subplots(len(urine_inds), 1)
    for i, (x, u) in enumerate(zip(x_inds, urine_inds)):
        urine_binary = form_design[:, u] > 0
        x_data = np.abs(form_design[:, x])
        prob, prior, bins = prob_over_x(x_data, urine_binary, 0, 62, bin_num=11)
        axs[i].plot(bins, prior)
        axs[i].plot(bins, prob)
    plt.show()


def run_regression_formation(exp_data: BasicExp):
    form_data = exp_data.filter_by_group('Territory/block', '0')
    all_data = make_full_design_matrix(form_data)
    all_data = all_data[:, :-1]
    all_data[:, 4] = all_data[:, 4] > 0
    all_data[:, 9] = all_data[:, 9] > 0
    urine_inds = [4, 9]
    # urine_inds = np.arange(0, 12)
    for u in urine_inds:
        this_des = np.delete(all_data, u, 1)
        # this_des = all_data
        urine_target = all_data[:, u]
        test_model = LogisticRegression(class_weight='balanced')
        y_test, pred, full_pred, r2, model = train_test_model(this_des, urine_target, model=test_model)
        f1 = f1_score(y_test, pred)
        print(f1, '...', r2)
        plt.plot(y_test + 1)
        plt.plot(pred)
        plt.show()


def run_regression_explore(exp_data: BasicExp):
    explore_data = exp_data.filter_by_group('block', '1')
    all_data = []
    for r in explore_data.runs:
        this_b0 = exp_data.filter_by_group('block', '0').filter_by_group('day', r.info['day']).filter_by_group('intruder_id', r.info['intruder_id'])
        if len(this_b0.runs) > 0:
            try:
                base_design = make_design_matrix(r.data, r.info, norm='0-1')
                my_urine = this_b0.runs[0].data[0]['urine_data']
                their_urine = this_b0.runs[0].data[1]['urine_data']
                dtmu = dist_to_urine(r.data[0]['x_cm'], r.data[0]['y_cm'], my_urine, thresh=20)
                dttu = dist_to_urine(r.data[0]['x_cm'], r.data[0]['y_cm'], their_urine, thresh=20)

                full_design = np.vstack((base_design, dtmu, dttu))
                all_data.append(full_design)
                print('cool')
            except IndexError:
                print('oof')
    all_data = np.hstack(all_data).T
    test_bs = [0, 1]
    target = all_data[:, test_bs]
    this_des = np.delete(all_data, test_bs, 1)
    # test_model = LogisticRegression(class_weight='balanced')
    test_model = LinearRegression()
    y_test, pred, full_pred, r2, model = train_test_model(this_des, target, model=test_model)
    # f1 = f1_score(y_test, pred)
    print(r2)
    plt.plot(y_test[:, 0] + 1)
    plt.plot(pred[:, 0])
    plt.show()


def plot_all_block0(exp_data: BasicExp, save_figs=False):
    block0 = exp_data.filter_by_group('Territory/block', '0')

    def plot_block0(run_data, run_info):
        try:
            design = make_design_matrix(run_data, run_info)
            design = design[:, :-1]
            f = plt.figure(layout='tight', figsize=(30, 20))
            gs = plt.GridSpec(4, 5)
            test_inds = [0, 1, 2, 3, 5, 6, 7, 8, 10]
            plot_x = [0, 0, 1, 1, 2, 2, 3, 3, 4]
            plot_y = [0, 2, 0, 2, 0, 2, 0, 2, 0]
            b_ids = ['MouseL_Dist_Wall (cm)', 'MouseL_Y (cm)', 'MouseL_Vel (cm/s)', 'MouseL_Ang (rad)',
                     'MouseR_Dist_Wall (cm)', 'MouseR_Y (cm)', 'MouseR_Vel (cm/s)', 'MouseR_Ang (rad)', 'Dist_Between_Mice (cm)']
            urine_raster_L = design[:, 4] > 0
            urine_raster_R = design[:, 9] > 0
            u_rasts = (urine_raster_L, urine_raster_R)
            cols = ['tab:blue', 'tab:orange']
            for px, py, name, ind in zip(plot_x, plot_y, b_ids, test_inds):
                this_x = design[:, ind]
                bin_max = np.max(this_x)
                if ind in [2, 6]:
                    bin_max = 20
                for i in range(2):
                    ax = f.add_subplot(gs[py + i, px])
                    probs, prior, bin_left_edge = prob_over_x(this_x, u_rasts[i], np.min(this_x), bin_max)
                    ax.plot(bin_left_edge, prior, color='k', label='P(x)')
                    ax.plot(bin_left_edge, probs, color=cols[i], label='P(pee|x)')
                    ax.set_xlabel(name)
                    plt.legend()

            traj_ax = f.add_subplot(gs[2:, 4])
            add_territory_circle(traj_ax, block='block0')
            urine_L = proj_urine_across_time(run_data[0]['urine_data'])
            urine_R = proj_urine_across_time(run_data[1]['urine_data'])
            traj_ax.scatter(urine_L[:, 0], urine_L[:, 1], marker='+', facecolor='g', zorder=10, label='urine+')
            traj_ax.scatter(urine_R[:, 0], urine_R[:, 1], marker='+', facecolor='g', zorder=10)
            traj_ax.plot(run_data[0]['x_cm'], run_data[0]['y_cm'])
            traj_ax.plot(run_data[1]['x_cm'], run_data[1]['y_cm'])
            plt.legend()
            sub = run_info['Subject']['subject_id']
            int = run_info['Territory']['intruder_id']
            day = run_info['Territory']['day']
            block = run_info['Territory']['block']
            run_name = fr'MouseL_{sub}_MouseR_{int}_{day}_Block{block}'
            traj_ax.set_title(run_name)
            file_name = fr'D:/TerritoryTools/plots/{run_name}.png'
            if save_figs:
                plt.savefig(file_name)
            else:
                plt.show()

        except IndexError:
            print('oof')

    block0.compute_across_runs(plot_block0, pass_info=True)


def fix_urine_output(root_dir):
    for f in os.listdir(root_dir):
        this_f = os.path.join(root_dir, f)
        t_files = find_territory_files(this_f)
        if t_files[-1] is not None:
            this_dict, this_md = import_all_data(this_f)
            orient = this_md['Territory']['orientation'].lower()
            if len(this_dict) == 1:
                urine = this_dict[0]['urine_data']
                rot_x, rot_y = rotate_xy(urine[:, 1], urine[:, 2], orient)
                urine[:, 1] = rot_x
                urine[:, 2] = rot_y
                this_dict[0]['urine_data'] = urine
            else:
                urine = np.vstack((this_dict[0]['urine_data'], this_dict[1]['urine_data']))
                rot_x, rot_y = rotate_xy(urine[:, 1], urine[:, 2], orient)
                urine[:, 1] = rot_x
                urine[:, 2] = rot_y
                this_dict[0]['urine_data'] = urine[urine[:, 1] < 0]
                this_dict[1]['urine_data'] = urine[urine[:, 1] > 0]
            save_path = t_files[-1].split('.')[0] + '_output.npy'
            np.save(save_path, this_dict, allow_pickle=True)


def urine_synchrony(exp_data: BasicExp):
    block0 = exp_data.filter_by_group('Territory/block', '0')
    all_data = make_full_design_matrix(block0, norm='0-1')
    m0_u = all_data[:, 4]
    m1_u = all_data[:, 9]
    # m0_u_start = np.argwhere(np.diff(m0_u) > 0.5)/40
    # m1_u_start = np.argwhere(np.diff(m1_u) > 0.5)/40
    # t = np.linspace(0, len(m0_u)/40, len(m0_u))
    # plt.plot(t, m0_u, label='MouseL')
    # plt.scatter(m0_u_start, np.ones_like(m0_u_start), label='MouseL Onsets')
    # plt.plot(t, m1_u + 1.1, label='MouseR')
    # plt.scatter(m1_u_start, 2.1*np.ones_like(m1_u_start), label='MouseR Onsets')
    # plt.xlabel('Time (s)')
    # plt.xlim(0, 15000)
    # plt.legend()
    # plt.show()
    # m0_on = np.hstack((0, (np.diff(m0_u) > 0.5).astype(int)))
    # m1_on = np.hstack((0, (np.diff(m1_u) > 0.5).astype(int)))
    # m0_cdf = np.cumsum(m0_on) / np.sum(m0_on)
    # m1_cdf = np.cumsum(m1_on) / np.sum(m1_on)
    # plt.plot(t, m0_cdf)
    # plt.plot(t, m1_cdf)
    # plt.show()

    # n_shuf = 1000
    # real_r = r_regression(m0_u[:, None], m1_u)
    # shuf_data = chunk_shuffle(m0_u, 600, num_shuffs=n_shuf)
    # r_null = r_regression(shuf_data.T, m1_u)
    # plt.hist(r_null)
    # plt.plot([real_r, real_r], [0, plt.ylim()[1]])
    # print(np.sum(r_null > real_r) / n_shuf)
    # shuf_r2s = np.zeros(n_shuf)
    # for i, s in enumerate(shuf_data):
    #     shuf_r2s[i] = r2


    print('y')
    # plt.show()


def urine_granger(exp_data: BasicExp, max_lag=400):
    block0 = exp_data.filter_by_group('Territory/block', '0')
    all_data = make_full_design_matrix(block0, norm='0-1')
    pee_rasts = (all_data[:, (4, 9)] > 0).astype(int)
    lags = np.arange(1, max_lag).astype(int)
    p_acc = []
    p_acc_r = []
    for lag in lags:
        grang_dict = grangercausalitytests(pee_rasts, [lag])
        p_val = grang_dict[lag][0]['ssr_ftest'][1]
        p_acc.append(p_val)
        grang_dict = grangercausalitytests(np.fliplr(pee_rasts), [lag])
        p_val = grang_dict[lag][0]['ssr_ftest'][1]
        p_acc_r.append(p_val)
    plt.plot(lags, p_acc)
    plt.plot(lags, p_acc_r)
    plt.show()


def plot_cc(run_data, run_info):
    f, ax = plt.subplots(2, 1)
    design = make_design_matrix(run_data, run_info, norm='zscore')
    ax[0].plot(design[:, 4])
    ax[0].plot(design[:, 9])
    corr = correlate(design[:, 4], design[:, 9], mode='full')
    corr_r = corr / len(design[:, 4])
    lags = correlation_lags(len(design[:, 9]), len(design[:, 4]))/40
    ax[1].plot(lags, corr_r)
    ax[1].set_xlabel('Lag (s)')
    ax[1].set_ylabel('r')
    ax[1].set_ylim(-0.1, 0.45)


def form_regression_temp(block0_exp):
    all_data = make_full_design_matrix(block0_exp, norm='0-1')
    m_data = all_data[:, 4]
    test_f = [400, 800, 1200, 2400, 4800]
    r2_acc = []
    for f in test_f:
        print(fr'On delay {f}')
        x_mat = trunc_hankel(m_data, f)
        model_out = train_test_model(x_mat, all_data[:, 9])
        r2 = model_out[3]
        r2_acc.append(r2)
    plt.plot(np.array(test_f)/40, r2_acc)
    plt.xlabel('Time Window (s)')
    plt.ylabel('R2')
    plt.show()


def trunc_hankel(time_series, buf_steps):
    sub_mat = np.zeros((len(time_series), buf_steps))
    for i in range(len(time_series)):
        pad = max(i - buf_steps, 0)
        this_chunk = time_series[pad:i]
        sub_mat[i, 0:len(this_chunk)] = this_chunk
    return sub_mat


def mi_block0(b0_run_data, b0_run_info):
    print(b0_run_info['Territory'])
    n_shuf = 1000
    design = make_design_matrix(b0_run_data, b0_run_info, norm='0-1')
    m0_u = (design[:, 4] > 0).astype(int)
    m1_u = (design[:, 9] > 0).astype(int)
    mi = mutual_info_regression(m0_u[:, None], m1_u)
    shuf_data = chunk_shuffle(m0_u, 600, num_shuffs=n_shuf)
    mi_null = mutual_info_regression(shuf_data.T, m1_u)
    pval = np.sum(mi < mi_null) / n_shuf
    null_mean = np.mean(mi_null)
    f, axs = plt.subplots(1, 2, figsize=(15, 10))
    title = fr'MI={mi}, mean_null={null_mean}, total_l={np.sum(m0_u)}, total_r={np.sum(m1_u)}, p_val={pval}'
    axs[0].hist(mi_null)
    axs[0].set_title(title)
    axs[0].plot([mi, mi], [0, plt.ylim()[1]], c='r')
    axs[0].plot([null_mean, null_mean], [0, plt.ylim()[1]], c='k')
    traj_ax = axs[1]
    add_territory_circle(traj_ax, block='block0')
    urine_L = proj_urine_across_time(b0_run_data[0]['urine_data'])
    urine_R = proj_urine_across_time(b0_run_data[1]['urine_data'])
    traj_ax.scatter(urine_L[:, 0], urine_L[:, 1], marker='+', facecolor='g', zorder=10, label='urine+')
    traj_ax.scatter(urine_R[:, 0], urine_R[:, 1], marker='+', facecolor='g', zorder=10)
    print(title)
    filename = b0_run_info['Territory']['intruder_id'] + b0_run_info['Territory']['orientation'] + b0_run_info['Subject']['subject_id'] + '.png'
    plt.savefig(filename)


if __name__ == '__main__':
    data_dir = 'D:/TerritoryTools/data/dataset'
    # make_save_design_matrix(data_dir)
    full_exp = load_dataset(data_dir)
    full_exp.compute_across_runs(post_clean_urine)
    block0 = full_exp.filter_by_group('Territory/block', '0')
    # f, cdf_axs = plt.subplots(1, 2)
    # full_exp.compute_across_group('Territory/block', '0', plot_cdfs, cdf_axs)
    # plt.show()
    # urine_prob_group(full_exp)
    # run_regression_explore(full_exp)
    # run_regression_formation(full_exp)
    # plot_all_block0(full_exp)
    # urine_granger(full_exp)
    # block0.compute_across_runs(plot_cc, pass_info=True)
    # plt.show()
    # urine_synchrony(full_exp)
    # form_regression_temp(block0)
    block0.compute_across_runs(mi_block0, pass_info=True)
    # data_path = 'D:\TerritoryTools\data\datasetv2\PPsync2_Day1_Subject_KinB1_Intruder_KinA3_Block_0_RNI_g'
    # run_dict_list, run_md = import_all_data(data_dir)
    # show_urine_segmented(run_dict_list[1]['urine_data'], run_dict_list[1]['urine_segment'])
    # plt.show()
    # mi_block0(run_dict_list, run_md)
