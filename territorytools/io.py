from territorytools.behavior import get_territory_data
from territorytools.urine import Peetector, sleap_to_fill_pts
import h5py
import cv2
import matplotlib.pyplot as plt
import numpy as np


def import_all_data(folder_name):
    """

    Parameters
    ----------
    folder_name - path which contains optical and thermal data. folder must contain the following files: optical.h5
    optical.mp4 thermal.h5 thermal.avi

    Returns
    -------
    Dict with keys 'mouse_xy' 'mouse_vel' 'mouse_heading' 'urine_times' and 'urine_xys'

    """

    op_vid = cv2.VideoCapture(folder_name + '/' + 'optical.mp4')
    was_read, first_frame = op_vid.read()
    f, ax = plt.subplots(1, 1)
    ax.imshow(first_frame)
    cal_line = plt.ginput(n=2, timeout=300, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
    op_cent_x = cal_line[0][0]
    op_cent_y = cal_line[0][1]
    sleap_file = h5py.File(folder_name + '/' + 'optical.h5', 'r')
    sleap_data = sleap_file['tracks']
    cent_x, cent_y, head_angles, vel, sleap_pts = get_territory_data(sleap_data, rot_offset=0, px_per_ft=350, ref_point=(op_cent_x, op_cent_y))
    fill_pts = sleap_to_fill_pts(folder_name + '/' + 'thermal.h5')
    peetect = Peetector(folder_name + '/' + 'thermal.avi', fill_pts, rot_ang=0)
    urine_t, urine_xys = peetect.peetect_frames(frame_win=20)
    # out_dict = {'mouse_xy': np.vstack((cent_x, cent_y)).T,
    #             'mouse_vel': vel,
    #             'mouse_heading': head_angles,
    #             'urine_times': ,
    #             'urine_xys': }
    return None