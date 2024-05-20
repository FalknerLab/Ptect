import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from territorytools.io import import_all_data
from territorytools.behavior import xy_to_territory
import cv2


def add_territory_circle(ax, block=None):
    circ = Circle((0, 0), radius=30.38, facecolor=(0.9, 0.9, 0.9), edgecolor=(0.2, 0.2, 0.2), linestyle='--')
    ax.add_patch(circ)
    if block == 'block0':
        ax.plot([0, 0], [0, -30.48], color=(0.2, 0.2, 0.2), linestyle='--')
        ax.plot([0, 30.48*np.sin(np.radians(60))], [0, 30.48*np.cos(np.radians(60))], color=(0.2, 0.2, 0.2), linestyle='--')
        ax.plot([0, 30.48 * np.sin(np.radians(-60))], [0, 30.48 * np.cos(np.radians(-60))], color=(0.2, 0.2, 0.2),
                linestyle='--')


def data_movie(optical_vid, therm_vid, run_dicts, block, num_frames=800, save_path=None):
    num_mice = len(run_dicts)
    f, axs = plt.subplots(1, 2, figsize=(11, 5))
    # op_vid = cv2.VideoCapture(optical_vid)
    t_vid = cv2.VideoCapture(therm_vid)
    t_vid.set(cv2.CAP_PROP_POS_FRAMES, 5)
    # o_im = axs[0].imshow(np.zeros((int(op_vid.get(4)), int(op_vid.get(3)))), aspect='auto')
    t_im = axs[0].imshow(np.zeros((int(t_vid.get(4)), int(t_vid.get(3)))))
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    ter_cols = ['tab:green', 'tab:orange', 'tab:gray']
    add_territory_circle(axs[1], block)

    arrows = []
    for i in range(num_mice):
        arrows.append(axs[1].plot([0, 0], [1, 1], color=ter_cols[i], marker='o', markevery=2, linewidth=4,
                                  label='Mouse: ' + str(i))[0])

    # u_time_cmap = plt.get_cmap('summer').resampled(num_frames)
    # u_time_colors = u_time_cmap(np.linspace(0, 1, num_frames))
    # plt.colorbar(cmap=u_time_cmap, ax=axs[2])
    u_scat = [axs[1].scatter([], [], facecolor='k', marker=',', s=1) for ind in range(num_mice)]
    axs[1].set_xlim(-32, 32)
    axs[1].set_ylim(-32, 32)
    axs[1].set_title('Output Data')
    axs[1].set_xlabel('X Position (cm)')
    axs[1].set_ylabel('Y Position (cm)')
    axs[1].legend()



    def show_frame(t):
        # _, op_f = op_vid.read()
        _, t_f = t_vid.read()

        # o_im.set_data(op_f)
        t_im.set_data(t_f[..., ::-1])
        for i, (m, a, u_s) in enumerate(zip(run_dicts, arrows, u_scat)):
            x = m['x_cm'][t]
            y = m['y_cm'][t]
            t_id = xy_to_territory(x, y)
            ang = m['angle'][t]
            a.set_xdata([x, x + 2*np.cos(ang)])
            a.set_ydata([y, y + 2*np.sin(ang)])
            a.set_markeredgecolor(ter_cols[t_id])
            if len(m['urine_data']) > 0:
                u_inds = m['urine_data'][:, 0] == t
                if sum(u_inds) > 0:
                    cur_pnts = np.vstack((u_s._offsets, m['urine_data'][u_inds, 1:]))
                    # new_pnts = np.unique(cur_pnts, axis=0)
                    # u_s._facecolors = np.vstack((u_s._facecolors, np.repeat(u_time_colors[t, :][:, None], sum(u_inds), axis=1).T))
                    u_s._offsets = cur_pnts
                    new_c = np.array(matplotlib.colors.to_rgba(ter_cols[i]))
                    new_c[:3] = (t/num_frames)*new_c[:3]
                    new_cols = np.repeat(new_c[:, None], sum(u_inds), axis=1).T
                    u_s._facecolors = np.vstack((u_s._facecolors, new_cols))

    anim = FuncAnimation(fig=f, func=show_frame, frames=num_frames, interval=40)
    anim.save(save_path, fps=80)


if __name__ == '__main__':
    run_dir = 'D:/TerritoryTools/demo/demo_run/'
    peetect_vid = 'D:/TerritoryTools/demo/peetect_demo.mp4'
    mouse_dicts = import_all_data(run_dir, num_mice=2, run_t_sec=120, block='block0', bypass_peetect=False,
                                  urine_output_vid_path=peetect_vid)
    print('Visualizing data...')
    data_movie(run_dir + 'demo_optical.mp4', peetect_vid, mouse_dicts, 'block0', num_frames=120*40-40,
               save_path='D:/TerritoryTools/demo/demo_out.mp4')


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