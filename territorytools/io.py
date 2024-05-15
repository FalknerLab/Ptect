import os
from territorytools.behavior import get_territory_data, interp_behavs
from territorytools.urine import Peetector, sleap_to_fill_pts, expand_urine_data, urine_segmentation, get_urine_source
import h5py
import numpy as np


def import_all_data(folder_name, num_mice=1, urine_frame_thresh=40, urine_heat_thresh=80, block=None,
                    urine_output_vid_path=None, show_all=True, start_t_sec=0, run_t_sec=None, samp_rate=40):
    """

    Parameters
    ----------
    folder_name - path which contains optical and thermal data. folder must contain the following files/suffixes:
    _optical.h5 _optical.mp4 _thermal.h5 _thermal.avi (or _thermal.mp4)
    where anything before the underscore is ignored

    Returns
    -------
    List[Dict] for each mouse with keys 'x_cm' 'y_cm' 'angle' 'velocity' 'urine_data'

    """

    slp_data = []
    slp_vid = []
    therm_data = []
    therm_vid = []
    for f in os.listdir(folder_name):
        f_splt = f.split('_')
        if f_splt[-1] == 'optical.h5':
            slp_data = f
        if f_splt[-1] == 'optical.mp4':
            slp_vid = f
        if f_splt[-1] == 'thermal.h5':
            therm_data = f
        if f_splt[-1] == 'thermal.avi' or f_splt[-1] == 'thermal.mp4':
            therm_vid = f

    start_f = int(start_t_sec*samp_rate)
    run_f = int(run_t_sec*samp_rate)

    print('Loading SLEAP data...')
    sleap_file = h5py.File(folder_name + '/' + slp_data, 'r')
    sleap_data = sleap_file['tracks']
    num_frames = sleap_data[0].shape[2]
    mice_cents = np.zeros((num_frames, 2, num_mice))
    for i in range(num_mice):
        cent_x, cent_y, head_angles, vel, sleap_pts = get_territory_data(sleap_data[i], rot_offset=0)
        interp_x, interp_y = interp_behavs(cent_x, cent_y)
        mice_cents[:, 0, i] = interp_x
        mice_cents[:, 1, i] = interp_y

    fill_pts = sleap_to_fill_pts(folder_name + '/' + therm_data)
    peetect = Peetector(folder_name + '/' + therm_vid, fill_pts)
    if block is not None:
        peetect.add_dz(zone=block)
    urine_data = peetect.peetect_frames(frame_win=urine_frame_thresh, save_vid=urine_output_vid_path, show_vid=False,
                                        start_frame=start_f, num_frames=run_f, heat_thresh=urine_heat_thresh)
    urine_seg = urine_segmentation(urine_data, do_animation=show_all)
    # urine_mouse = get_urine_source(mice_cents, exp_urine)

    mouse_list = []
    for m in range(num_mice):
        out_dict = {'x_cm': mice_cents[:, 0, m],
                    'y_cm': mice_cents[:, 1, m],
                    'urine_data': urine_data}
        mouse_list.append(out_dict)

    return mouse_list
