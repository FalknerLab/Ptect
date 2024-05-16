

def plot_all_urine_xys(g_id, g_data, g_info, params):
    c, ax = params[:]
    group_n = len(g_data)
    for g in g_data:
        ax.scatter(g[:, 0], g[:, 1], color=c, s=1)
    ax.set_title('Group: ' + g_id)
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    return None


def plot_block0(run_data):
    times, evt_xys = run_data[:]
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    gs = GridSpec(2, 4, figure=fig)
    total_marks_left, total_marks_right = urine_area_over_time(run_data)
    times = np.arange(len(total_marks_left)) / (40 * 60)
    y1 = max(np.max(total_marks_left), np.max(total_marks_right))
    y1 = y1 + 0.2 * y1
    y1 = 1200
    ax0 = fig.add_subplot(gs[1, 2:])
    ax0.plot(times, total_marks_right, c=[1, 0.5, 0])
    ax0.set_ylim(0, y1)
    ax0.set_xlabel('Time (min)')
    ax0.set_ylabel('Urine Area (px)')
    ax1 = fig.add_subplot(gs[0, 2:])
    ax1.plot(times, total_marks_left, c=[0, 0.8, 0])
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Urine Area (px)')
    ax1.set_ylim(0, y1)
    _, mask_l, mask_r = get_mask(run_data)
    mask_h, mask_w = np.shape(mask_l)
    rgb = np.zeros((mask_h, mask_w, 3))
    rgb[:, :, 1] += mask_l
    rgb[:, :, 0] += mask_r
    rgb[:, :, 1] += mask_r / 2
    ax2 = fig.add_subplot(gs[:, :2])
    ax2.imshow(np.flipud(rgb))
    plt.show()

def plot_all_urine_xys(g_id, g_data, g_info, params):
    c, ax = params[:]
    group_n = len(g_data)
    for g in g_data:
        ax.scatter(g[:, 0], g[:, 1], color=c, s=1)
    ax.set_title('Group: ' + g_id)
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    return None

def plot_all_masks(g_id, g_data, g_info):
    f, axs = plt.subplots(1, 1, figsize=(10, 10))
    m0 = g_data[0]
    mask = np.zeros_like(m0)
    for g in g_data:
        mask += g
    axs.imshow(mask)
    axs.set_title(g_id)

def plot_all_series_and_mean(g_id, g_data, g_info):
    f, axs = plt.subplots(2, 1, figsize=(20, 10))
    m0 = g_data[0]
    num_f = len(m0[0])
    temp = np.zeros((num_f, len(g_data) * 2))
    c = -1
    for m in g_data:
        c += 1
        t = np.arange(num_f) / (40 * 60)
        axs[0].plot(t, m[0], c=[0.5, 0.5, 0.5])
        axs[0].plot(t, m[1], c=[0.5, 0.5, 0.5])
        temp[:, c] = m[0]
        temp[:, c + 1] = m[1]
    axs[1].plot(t, np.mean(temp, axis=1), c='r')
    plt.show()
    print('yes')

def run_cc(g_id, g_data, g_info):
    for g, g_i in zip(g_data, g_info):
        u_r, u_i = urine_area_over_time(g)
        corr = correlate(u_r, u_i)
        lags = correlation_lags(len(u_r), len(u_i))
        corr /= np.max(corr)
        plt.figure()
        plt.plot(lags / (40 * 60), corr)
        f_name = g_i['Resident'] + ' vs ' + g_i['Intruder'] + ' Day ' + g_i['Day']
        plt.title(f_name)
        plt.show()



# def plot_territory(ax, t, r):
#     walls = [-np.pi / 2, 5 * np.pi / 6, np.pi / 6]
#     num_f = len(t)
#     ter_a = np.logical_or(t < walls[0], t > walls[1])
#     ter_b = np.logical_and(t > walls[0], t < walls[2])
#     ter_c = np.logical_and(t > walls[2], t < walls[1])
#     cmap = np.tile([0.6, 0.6, 0.6], (num_f, 1))
#     ax.set_xticks([])
#     ax.set_yticks([])
#     cmap[ter_b, :] = [0, 0.6, 0]
#     cmap[ter_a, :] = [0.7, 0.6, 0.3]
#     # ax.scatter(t, r, c=cmap, s=3)
#     h, te, re, _ = ax.hist2d(t, r, bins=50, range=[[np.nanmin(t), np.nanmax(t)], [np.nanmin(r), np.nanmax(r)]],
#                              density=True)
#     return h, te, re

# def behavioral_raster(rasters, ax=plt.gca, fs=1):
#     cols = ['tab:green', 'tab:orange', 'tab:gray', 'c', 'm', 'r', 'k']
#     len_behav, num_rasts = np.shape(rasters)
#     rast_im = 255*np.ones((1, len_behav, 3))
#     for i in range(num_rasts):
#         col = to_rgb(cols[i])
#         inds = np.where(rasters[:, i])[0]
#         rast_im[:, inds, :] = col
#     ax.imshow(rast_im, aspect='auto')
#     ax.set_xlim([0, len_behav])
#     ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
#     x_ticks = ax.get_xticks()
#     x_ticks = x_ticks[x_ticks < len_behav]
#     ax.set_xticks(x_ticks, labels=x_ticks/fs)
#     ax.get_yaxis().set_visible(False)

# def hist_t(g, ts, info):
#     group_t = np.nan*np.zeros((len(ts), len(ts[0])))
#     count_acc = []
#     for ind, t in enumerate(ts):
#         group_t[ind, :] = t
#         vals, bin_edges = np.histogram(group_t, bins=36, range=[-np.pi, np.pi])
#         norm_vals = 100*(vals/sum(vals))
#         count_acc.append(norm_vals)
#
#     count_acc = np.array(count_acc)
#     mean_vals = np.mean(count_acc, axis=0)
#     dev = np.std(count_acc, axis=0)
#     n = np.shape(count_acc)[0]
#     sem = 1.96*(dev/np.sqrt(n))
#     ax = plt.subplot(projection='polar')
#     r_edge = 1.2*max(mean_vals)
#     pts = [-np.pi/2, 5*np.pi/6, np.pi/6]
#     for p in pts:
#         ax.plot([0, p], [0, r_edge], ':k', linewidth=2)
#
#     # cl = get_chance_line(group_t)
#     cl = 100*np.ones(len(mean_vals))/len(mean_vals)
#     vals = np.hstack((mean_vals, mean_vals[0]))
#     sem = np.hstack((sem, sem[0]))
#     cl = np.hstack((cl, cl[0]))
#     ax.plot(bin_edges, cl, color=[0, 0, 0], linewidth=2)
#     ax.plot(np.linspace(np.pi/6, 5*np.pi/6, 120), r_edge*np.ones(120), color=[0.5, 0.5, 0.5], linewidth=2)
#     ax.plot(np.linspace(5*np.pi/6, 1.5*np.pi, 120), r_edge*np.ones(120), color=[0.2, 0.6, 0.2], linewidth=2)
#     ax.plot(np.linspace(np.pi/6, -np.pi/2, 120), r_edge*np.ones(120), color=[0.8, 0.5, 0.2], linewidth=2)
#     ax.plot(bin_edges, vals + sem, color=[0.4, 0.4, 0.8], linewidth=2, linestyle='--')
#     ax.plot(bin_edges, vals - sem, color=[0.4, 0.4, 0.8], linewidth=2, linestyle='--')
#     ax.plot(bin_edges, vals, color=[0, 0, 0.8], linewidth=4)
#     ax.set_rlabel_position(0)
#     ax.set_xticklabels([])
#     ax.set_yticks(ax.get_yticks()[:-1])
#     ax.grid(axis="x")
#     plt.show()
# def get_chance_line(group_ts):
#     np.random.seed(5)
#     group_ts = np.degrees(group_ts)
#     n = np.shape(group_ts)[0]
#     ang_options = np.arange(-170, 170, 10)
#     hist_acc = []
#     num_i = 100
#     for i in range(num_i):
#         ang_inds = np.random.randint(0, len(ang_options), n)
#         these_angs = ang_options[ang_inds]
#         shift_ts = group_ts + np.expand_dims(these_angs, axis=1)
#         fix_inds = np.where(shift_ts > 180)
#         shift_ts[fix_inds] = shift_ts[fix_inds] - 360
#         fix_inds1 = np.where(shift_ts < -180)
#         shift_ts[fix_inds1] = shift_ts[fix_inds1] + 360
#         shift_ts = np.radians(shift_ts)
#         for j in range(n):
#             vals, bin_edges = np.histogram(shift_ts[j, :], bins=36, range=[-np.pi, np.pi])
#             norm_vals = vals/sum(vals)
#             hist_acc.append(norm_vals)
#     out_acc = np.array(hist_acc)
#     chance_line = 100*np.mean(out_acc, axis=0)
#     return chance_line

def plot_prefs_across_group(g, group_data, group_info, ax, ids):
    pref_mat = np.array(group_data)
    marker_list = list(Line2D.markers.keys())
    marker_ind = np.argwhere(np.equal(ids, g))[0][0]
    mark = marker_list[marker_ind]
    ax.plot(pref_mat[:, 0], marker=mark, color='g')
    ax.plot(pref_mat[:, 1], marker=mark, color=[0.7, 0.3, 0])
    ax.plot(pref_mat[:, 2], marker=mark, color=[0.3, 0.3, 0.3])


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