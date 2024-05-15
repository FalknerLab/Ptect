import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from territorytools.io import import_all_data
import cv2


def data_movie(optical_vid, therm_vid, run_dicts):

    f, axs = plt.subplots(3, 1)
    op_vid = cv2.VideoCapture(optical_vid)
    t_vid = cv2.VideoCapture(therm_vid)
    t_vid.set(cv2.CAP_PROP_POS_FRAMES, 5)
    o_im = axs[0].imshow(np.zeros((int(op_vid.get(4)), int(op_vid.get(3)))))
    t_im = axs[1].imshow(np.zeros((int(t_vid.get(4)), int(t_vid.get(3)))))
    s = axs[2].scatter([], [])
    axs[2].set_xlim(-32, 32)
    axs[2].set_ylim(-32, 32)

    def show_frame(t):
        _, op_f = op_vid.read()
        _, t_f = t_vid.read()
        o_im.set_data(op_f)
        t_im.set_data(t_f)
        for i, m in enumerate(run_dicts):
            s._offsets[i] = m['mouse_xy'][t]

    anim = FuncAnimation(fig=f, func=show_frame, frames=1200, interval=10)
    plt.show()


if __name__ == '__main__':
    run_dir = 'D:/TerritoryTools/tests/test_run/'
    mouse_dicts = import_all_data(run_dir, num_mice=2)
    # data_movie(run_dir + 'demo_optical.mp4',
    #            run_dir + 'demo_thermal.avi', mouse_dicts)

