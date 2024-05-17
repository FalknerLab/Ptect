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
    mouse_dicts = import_all_data(run_dir, num_mice=2, run_t_sec=60, block='block0', bypass_peetect=False,
                                  urine_output_vid_path=peetect_vid)
    print('Visualizing data...')
    data_movie(run_dir + 'demo_optical.mp4', peetect_vid, mouse_dicts, 'block0', num_frames=60*40,
               save_path='D:/TerritoryTools/demo/demo_out.mp4')
