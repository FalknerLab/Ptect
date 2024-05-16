import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from territorytools.io import import_all_data
import cv2


def add_territory_circle(ax, block=None):
    circ = Circle((0, 0), radius=30.38, facecolor=(0.9, 0.9, 0.9), edgecolor=(0.2, 0.2, 0.2), linestyle='--')
    ax.add_patch(circ)
    if block == 'block0':
        ax.plot([0, 0], [0, -30.48], color=(0.2, 0.2, 0.2), linestyle='--')
        ax.plot([0, 30.48*np.sin(np.radians(60))], [0, 30.48*np.cos(np.radians(60))], color=(0.2, 0.2, 0.2), linestyle='--')
        ax.plot([0, 30.48 * np.sin(np.radians(-60))], [0, 30.48 * np.cos(np.radians(-60))], color=(0.2, 0.2, 0.2),
                linestyle='--')


def data_movie(optical_vid, therm_vid, run_dicts, block):

    f, axs = plt.subplots(1, 3, figsize=(20, 8))
    op_vid = cv2.VideoCapture(optical_vid)
    t_vid = cv2.VideoCapture(therm_vid)
    t_vid.set(cv2.CAP_PROP_POS_FRAMES, 5)
    o_im = axs[0].imshow(np.zeros((int(op_vid.get(4)), int(op_vid.get(3)))), aspect='auto')
    t_im = axs[1].imshow(np.zeros((int(t_vid.get(4)), int(t_vid.get(3)))), aspect='auto')
    add_territory_circle(axs[2], block)
    scats = [axs[2].scatter(0, 0) for d in run_dicts]
    u_scat = [axs[2].scatter([], [], c=c, marker=',') for c in ['r', 'g']]
    axs[2].set_xlim(-32, 32)
    axs[2].set_ylim(-32, 32)

    def show_frame(t):
        _, op_f = op_vid.read()
        _, t_f = t_vid.read()
        o_im.set_data(op_f)
        t_im.set_data(t_f)
        for i, (m, s, u_s) in enumerate(zip(run_dicts, scats, u_scat)):
            s._offsets = [[m['x_cm'][t], m['y_cm'][t]]]
            if len(m['urine_data']) > 0:
                u_inds = m['urine_data'][:, 0] == t
                cur_pnts = np.vstack((u_s._offsets, m['urine_data'][u_inds, 1:]))
                new_pnts = np.unique(cur_pnts, axis=0)
                u_s._offsets = new_pnts

    anim = FuncAnimation(fig=f, func=show_frame, frames=2400, interval=40)
    anim.save('demo_out.mp4', fps=80)


if __name__ == '__main__':
    run_dir = 'D:/TerritoryTools/demo/demo_run/'
    mouse_dicts = import_all_data(run_dir, num_mice=2, run_t_sec=10, block='block0')
    data_movie(run_dir + 'demo_optical.mp4',
               run_dir + 'demo_thermal.mp4', mouse_dicts, 'block0')


