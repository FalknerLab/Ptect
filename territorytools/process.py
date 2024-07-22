import os
import yaml
import shutil
import h5py
import numpy as np
from behavior import get_territory_data, interp_behavs, compute_preferences, make_design_matrix, rotate_xy
from urine import Peetector, urine_segmentation


def import_all_data(folder_name, urine_time_thresh=1, urine_heat_thresh=110, show_all=True, start_t_sec=0, samp_rate=40):
    """

    Parameters
    ----------
    folder_name - path which contains optical and thermal data. folder must contain the following files/suffixes:
    _top.h5 _top.mp4 _thermal.h5 _thermal.avi
    where anything before the underscore is ignored

    Returns
    -------
    List[Dict] for each mouse with keys 'x_cm' 'y_cm' 'angle' 'velocity' 'urine_data'

    """

    if not valid_dir(folder_name):
        raise Exception('Folder does not contain all territory data files (top.mp4, top.h5, thermal.avi, thermal.h5)')

    top_vid, top_data, therm_vid, therm_data, fixed_top, fixed_therm, md_path, pt_vid, pt_data, out_data = find_territory_files(folder_name)
    md_dict = yaml.safe_load(open(md_path))

    if out_data is not None:
        run_data = np.load(out_data, allow_pickle=True)
        return run_data, md_dict

    orient = md_dict['Territory']['orientation'].lower()
    block = int(md_dict['Territory']['block'])

    if fixed_top is None:
        fixed_top = clean_sleap_h5(top_data, block=block, orientation=orient)

    if fixed_therm is None:
        fixed_therm = clean_sleap_h5(therm_data, block=block, orientation=orient, cent_xy=(320, 210))

    print('Loading SLEAP data...')
    sleap_file = h5py.File(fixed_top, 'r')
    sleap_data = sleap_file['tracks']

    num_frames = sleap_data[0].shape[2]
    num_mice = sleap_data.shape[0]
    mice_cents = np.zeros((num_frames, 2, num_mice))
    angs = []
    vels = []
    t_ids = []
    cent_x = None
    for i in range(num_mice):
        cent_x, cent_y, head_angles, vel, sleap_pts = get_territory_data(sleap_data[i], rot_offset=orient)
        _, t_id = compute_preferences(cent_x, cent_y)
        interp_x, interp_y, interp_angs, interp_vel = interp_behavs(cent_x, cent_y, head_angles, vel)
        mice_cents[:, 0, i] = interp_x
        mice_cents[:, 1, i] = interp_y
        angs.append(interp_angs)
        vels.append(interp_vel)
        t_ids.append(t_id)

    start_f = int(start_t_sec * samp_rate)

    run_f = len(cent_x)
    urine_seg = []
    urine_mouse = []
    if out_data is None:
        peetect = Peetector(therm_vid, fixed_therm, heat_thresh=urine_heat_thresh)
        peetect.add_dz(zone=fr'block{block}')
        f_parts = os.path.split(folder_name)[-1]
        pt_vid_path = os.path.join(folder_name, f_parts + '_ptvid.mp4')
        urine_data = peetect.peetect_frames(time_thresh=urine_time_thresh, save_vid=pt_vid_path, show_vid=show_all,
                                            start_frame=start_f, num_frames=None)
        if len(urine_data) > 0:
            urine_seg = urine_segmentation(urine_data)
            urine_mouse = np.zeros_like(urine_data[:, 1])
            if not block:
                urine_mouse = (urine_data[:, 1] > 0).astype(int)

    mouse_list = []
    for m in range(num_mice):
        m_urine = []
        clus_id = []
        if len(urine_mouse) > 0:
            m_urine_inds = urine_mouse == m
            m_urine = urine_data[m_urine_inds, :]
            m_urine_seg = urine_seg[m_urine_inds]
            _, clus_id = np.unique(m_urine_seg, return_inverse=True)
        out_dict = {'x_cm': mice_cents[:, 0, m],
                    'y_cm': mice_cents[:, 1, m],
                    'angle': angs[m],
                    'velocity': vels[m],
                    'ter_id': t_ids[m],
                    'urine_data': m_urine,
                    'urine_segment': clus_id}
        mouse_list.append(out_dict)

    if out_data is None:
        f_parts = os.path.split(folder_name)[-1]
        save_path = os.path.join(folder_name, f_parts + '_output.npy')
        np.save(save_path, mouse_list, allow_pickle=True)

    return mouse_list, md_dict


def find_territory_files(root_dir: str):
    top_data = None
    top_vid = None
    therm_data = None
    therm_vid = None
    fixed_top = None
    fixed_therm = None
    metadata_file = None
    peetect_vid = None
    peetect_data = None
    out_data = None
    for f in os.listdir(root_dir):
        f_splt = f.split('_')
        if f_splt[-1] == 'metadata.yaml':
            metadata_file = os.path.join(root_dir, f)
        if f_splt[-1] == 'top.h5':
            top_data = os.path.join(root_dir, f)
        if f_splt[-1] == 'top.mp4':
            top_vid = os.path.join(root_dir, f)
        if f_splt[-1] == 'thermal.h5':
            therm_data = os.path.join(root_dir, f)
        if f_splt[-1] == 'thermal.avi' or f_splt[-1] == 'thermal.mp4':
            therm_vid = os.path.join(root_dir, f)
        if f_splt[-1] == 'fixed.h5':
            if f_splt[-2] == 'thermal':
                therm_data = os.path.join(root_dir, f)
                fixed_therm = os.path.join(root_dir, f)
            if f_splt[-2] == 'top':
                top_data = os.path.join(root_dir, f)
                fixed_top = os.path.join(root_dir, f)
        if f_splt[-1] == 'ptvid.mp4':
            peetect_vid = os.path.join(root_dir, f)
        if f_splt[-1] == 'ptdata.npy':
            peetect_data = os.path.join(root_dir, f)
        if '.npy' in f_splt[-1]:
            out_data = os.path.join(root_dir, f)
    return top_vid, top_data, therm_vid, therm_data, fixed_top, fixed_therm, metadata_file, peetect_vid, peetect_data, out_data


def valid_dir(root_dir: str):
    files = os.listdir(root_dir)
    target_sufs = ['thermal.avi', 'thermal.h5', 'top.mp4', 'top.h5', 'metadata.yaml']
    contains_f = np.zeros_like(target_sufs) == 1
    for ind, t in enumerate(target_sufs):
        has_t = False
        for f in files:
            if f.split('_')[-1] == t:
                has_t = True
        contains_f[ind] = has_t
    return np.all(contains_f)


def convert_npy_h5(npy_path):
    path_parts = os.path.split(npy_path)
    new_file = path_parts[1].split('.')[0] + '_output.h5'
    out_h5 = os.path.join(path_parts[0], new_file)
    h5_file = h5py.File(out_h5, 'w')
    data_dict = np.load(npy_path, allow_pickle=True)
    for i, d in enumerate(data_dict):
        g = fr'Mouse{i}'
        h5_file.create_group(g)
        for k in d.keys():
            h5_file.create_dataset(g + '/' + k, data=d[k])
    h5_file.close()


def clean_sleap_h5(slp_h5: str, block=1, orientation=0, cent_xy=(638, 504)):
    rot_dict = {'rni': 0,
                'none': 0,
                'irn': 120,
                'nir': 240}
    rot_ang = orientation
    if type(orientation) is str:
        rot_ang = rot_dict[orientation]
    new_file = slp_h5.split('.')[0] + '_fixed.h5'
    # fixed_file = shutil.copy(slp_h5, new_file)
    slp_data = h5py.File(slp_h5, 'r')
    fixed_file = h5py.File(new_file, 'a')
    tracks = slp_data['tracks'][:]
    t_scores = slp_data['tracking_scores'][:]
    i_scores = slp_data['instance_scores'][:]
    p_scores = slp_data['point_scores'][:]
    t_occupancy = slp_data['track_occupancy'][:]
    len_t = np.shape(tracks)[-1]
    if block:
        out_name = slp_data['track_names'][0][:]
        out_ts = tracks[0][None, :]
        last_cent = np.nanmean(out_ts[0, :, :, 0], axis=1)
        for i in range(len_t):
            if i % 1000 == 0:
                print('Cleaning slp file, on frame: ', i)
            this_cent = np.nanmean(tracks[:, :, :, i], axis=2)
            dists = np.linalg.norm(this_cent - last_cent, axis=1)
            if not np.all(np.isnan(dists)):
                best_track = np.nanargmin(dists)
                t_scores[0, i] = t_scores[best_track, i]
                p_scores[0, :, i] = p_scores[best_track, :, i]
                i_scores[0, i] = i_scores[best_track, i]
                last_cent = this_cent[best_track, :]
                t_occupancy[i, 0] = t_occupancy[i, best_track]
                out_ts[0, :, :, i] = tracks[best_track, :, :, i]
        fixed_file['track_names'] = [out_name]
        fixed_file['tracking_scores'] = t_scores[0, :]
        fixed_file['instance_scores'] = i_scores[0, :]
        fixed_file['point_scores'] = p_scores[0, :, :]
        fixed_file['track_occupancy'] = t_occupancy[:, 0]
        fixed_file['tracks'] = out_ts
    else:
        out_name = slp_data['track_names'][:2][:]
        out_ts = np.nan * np.zeros_like(tracks[:2, :, :, :])
        for i in range(len_t):
            if i % 1000 == 0:
                print('Cleaning slp file, on frame: ', i)
            this_cent = np.nanmean(tracks[:, :, :, i], axis=2)
            rel_y = this_cent[:, 0] - cent_xy[0]
            rel_x = this_cent[:, 1] - cent_xy[1]
            rot_x, rot_y = rotate_xy(rel_x, rel_y, rot_ang)
            as_t = np.degrees(np.arctan2(rot_y, rot_x))
            rot_t = as_t
            in_res = rot_t < 0
            in_int = rot_t > 0
            if np.any(in_res):
                res_ind = np.where(in_res)[0][0]
                t_scores[0, i] = t_scores[res_ind, i]
                p_scores[0, :, i] = p_scores[res_ind, :, i]
                i_scores[0, i] = i_scores[res_ind, i]
                t_occupancy[i, 0] = 1
                out_ts[0, :, :, i] = tracks[res_ind, :, :, i]
            else:
                t_scores[0, i] = np.nan
                p_scores[0, :, i] = np.ones_like(p_scores[0, :, i]) * np.nan
                i_scores[0, i] = np.nan
                t_occupancy[i, 0] = 0
                out_ts[0, :, :, i] = np.ones_like(out_ts[0, :, :, 0]) * np.nan
            if np.any(in_int):
                int_ind = np.where(in_int)[0][0]
                t_scores[1, i] = t_scores[int_ind, i]
                p_scores[1, :, i] = p_scores[int_ind, :, i]
                i_scores[1, i] = i_scores[int_ind, i]
                t_occupancy[i, 1] = 1
                out_ts[1, :, :, i] = tracks[int_ind, :, :, i]
            else:
                t_scores[1, i] = np.nan
                p_scores[1, :, i] = np.ones_like(p_scores[1, :, i]) * np.nan
                i_scores[1, i] = np.nan
                t_occupancy[i, 1] = 0
                out_ts[1, :, :, i] = np.ones_like(out_ts[1, :, :, 0]) * np.nan
        fixed_file['track_names'] = out_name
        fixed_file['tracking_scores'] = t_scores[:2, :]
        fixed_file['instance_scores'] = i_scores[:2, :]
        fixed_file['point_scores'] = p_scores[:2, :, :]
        fixed_file['track_occupancy'] = t_occupancy[:, :2]
        fixed_file['tracks'] = out_ts
    fixed_file['edge_inds'] = slp_data['edge_inds'][:]
    fixed_file['edge_names'] = slp_data['edge_names'][:]
    fixed_file['labels_path'] = ()
    fixed_file['node_names'] = slp_data['node_names'][:]
    fixed_file['provenance'] = ()
    fixed_file['video_ind'] = ()
    fixed_file['video_path'] = ()
    return new_file


def package_data(root_dir):
    files = os.listdir(root_dir)
    for f in files:
        f_parts_u = f.split('_')
        run_dir = os.path.join(root_dir, '_'.join(f_parts_u[:-1]))
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        shutil.move(os.path.join(root_dir, f), run_dir)
        

def make_nwb_yamls(root_dir, key_fmt, key_del='_', val_del='_'):
    for f in os.listdir(root_dir):
        merge_dict = {}
        if os.path.isdir(os.path.join(root_dir, f)):
            append_dict = make_dict(key_fmt, f, key_del=key_del, val_del=val_del)
            yaml_file = os.path.join(root_dir, f, fr'{f}_metadata.yaml')
            shutil.copy('nwb_metadata.yaml', yaml_file)
            yaml_obj = open(yaml_file, 'r')
            yaml_dict = yaml.safe_load(yaml_obj)
            unique_keys = np.unique(list(yaml_dict.keys()) + list(append_dict.keys())).astype(list)
            for k in unique_keys:
                if k in yaml_dict.keys() and k in append_dict.keys():
                    merge_dict[k] = yaml_dict[k].copy()
                    merge_dict[k].update(append_dict[k])
                elif k in yaml_dict.keys():
                    merge_dict[k] = yaml_dict[k]
                elif k in append_dict.keys():
                    merge_dict[k] = append_dict[k]
            yaml_obj = open(yaml_file, 'w')
            yaml.safe_dump(merge_dict, yaml_obj)


def make_dict(key_fmt, val_fmt, key_del='_', val_del='_'):
    keys = key_fmt.split(key_del)
    vals = val_fmt.split(val_del)
    out_dict = {}
    for i, k in enumerate(keys):
        if k != '{}':
            nest_dict = k.split(':')
            if len(nest_dict) > 1:
                sub_dict = {}
                for ind in range(len(nest_dict), 0, -1):
                    if ind == len(nest_dict):
                        sub_dict[nest_dict[ind-1]] = vals[i]
                    else:
                        if nest_dict[ind-1] in out_dict.keys():
                            out_dict[nest_dict[ind-1]].update(sub_dict)
                        else:
                            out_dict[nest_dict[ind - 1]] = sub_dict
            else:
                out_dict[k] = vals[i]
    return out_dict


def make_save_design_matrix(root_dir):
    for f in os.listdir(root_dir):
        this_path = os.path.join(root_dir, f)
        out_name = os.path.join(this_path, f + '_design.h5')
        if valid_dir(this_path):
            this_run, this_info = import_all_data(this_path)
            try:
                design_mat = make_design_matrix(this_run, this_info['Territory'])
                out_file = h5py.File(out_name, 'w')
                out_file.create_dataset('design', data=design_mat)
                out_file.close()
            except IndexError:
                print(out_name)