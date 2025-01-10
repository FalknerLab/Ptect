import os
import yaml
import shutil
import h5py
import numpy as np

from territorytools.behavior import get_territory_data, interp_behavs, compute_preferences, make_design_matrix
from territorytools.utils import rotate_xy
from territorytools.urine import Peetector, urine_segmentation


def process_all_data(run_folder_root, show_all=False, start_t_sec=0, skip_ptect=True):

    if valid_dir(run_folder_root):
        print(run_folder_root)
    else:
        print(f'{run_folder_root} does not contain all territory dataset files (ptmetadata.yml, top.mp4, top.h5, thermal.mp4, thermal.h5)')
        return None

    files_dict = find_territory_files(run_folder_root)

    md_path = find_file_recur(run_folder_root, 'ptmetadata.yml')
    md_dict = yaml.safe_load(open(md_path, 'r'))

    folder_name = files_dict['root']
    num_mice = md_dict['Territory']['num_mice']
    optical_hz = md_dict['Territory']['optical_hz']
    thermal_hz = md_dict['Territory']['thermal_hz']
    thermal_px_per_cm = md_dict['Territory']['thermal_px_per_cm']
    urine_time_thresh = md_dict['Territory']['ptect_time_thresh']
    urine_heat_thresh = md_dict['Territory']['ptect_heat_thresh']
    optical_px_per_cm = md_dict['Territory']['optical_px_per_cm']
    optical_center = md_dict['Territory']['optical_center']
    urine_cool_thresh = md_dict['Territory']['ptect_cool_thresh']
    ptect_smooth_kern = md_dict['Territory']['ptect_smooth_kern']
    ptect_dilate_kern = md_dict['Territory']['ptect_dilate_kern']

    out_data = files_dict['output.npy']

    if out_data is not None:
        run_data = np.load(out_data, allow_pickle=True)
        return run_data

    orient = md_dict['Territory']['orientation'].lower()
    op_cent = md_dict['Territory']['optical_center']
    therm_cent = md_dict['Territory']['thermal_center']

    fixed_top = files_dict['fixedtop.h5']

    if files_dict['fixedtop.h5'] is None:
        fixed_top = clean_sleap_h5(files_dict['top.h5'], num_mice=num_mice, orientation=orient, suff='fixedtop', cent_xy=op_cent)

    therm = files_dict['thermal.h5']

    print('Loading SLEAP dataset...')
    sleap_file = h5py.File(fixed_top, 'r')
    sleap_data = sleap_file['tracks']

    num_frames = sleap_data[0].shape[2]
    num_mice = sleap_data.shape[0]
    mice_cents = np.zeros((num_frames, 2, num_mice))
    angs = []
    vels = []
    t_ids = []
    for i in range(num_mice):
        cent_x, cent_y, head_angles, vel, sleap_pts = get_territory_data(sleap_data[i],
                                                                         rot_offset=orient,
                                                                         px_per_cm=optical_px_per_cm,
                                                                         ref_point=optical_center,
                                                                         hz=optical_hz)
        _, t_id = compute_preferences(cent_x, cent_y)
        interp_x, interp_y, interp_angs, interp_vel = interp_behavs(cent_x, cent_y, head_angles, vel)
        mice_cents[:, 0, i] = interp_x
        mice_cents[:, 1, i] = interp_y
        angs.append(interp_angs)
        vels.append(interp_vel)
        t_ids.append(t_id)

    urine_seg = []
    urine_mouse = []
    urine_data = []
    ptect_data = files_dict['ptect.npz']

    if ptect_data is not None:
        ptect = np.load(ptect_data, allow_pickle=True)
        urine_data = ptect['hot_data']

    if ptect_data is None and not skip_ptect:
        peetect = Peetector(files_dict['thermal.avi'], therm,
                            hot_thresh=urine_heat_thresh,
                            cold_thresh=urine_cool_thresh,
                            cent_xy=therm_cent,
                            px_per_cm=thermal_px_per_cm,
                            hz=thermal_hz,
                            s_kern=ptect_smooth_kern,
                            di_kern=ptect_dilate_kern)
        # peetect.add_dz(zone=fr'block{block}')
        f_parts = os.path.split(folder_name)[-1]
        pt_vid_path = os.path.join(folder_name, f_parts + '_ptvid.mp4')
        urine_data = peetect.peetect_frames(save_vid=pt_vid_path,
                                            show_vid=show_all,
                                            start_frame=int(start_t_sec*thermal_hz))
    if len(urine_data) > 0:
        # urine_seg = urine_segmentation(urine_data)
        urine_seg = []
        urine_mouse = np.zeros_like(urine_data[:, 1])
        if num_mice > 1:
            urine_mouse = (urine_data[:, 1] > 0).astype(int)

    kpms_data = None
    if 'kpms_csv' in files_dict.keys():
        this_k = '_'.join(os.path.split(files_dict['fixedtop.h5'])[1].split('_')[:-1])
        kpms_dict = load_kpms(files_dict['kpms_csv'])
        vec_len = len(interp_x)
        this_syl = kpms_dict[this_k]
        out_syl = np.zeros_like(interp_x)
        out_syl[:len(this_syl)] = this_syl
        kpms_data = out_syl


    mouse_list = []
    for m in range(num_mice):
        m_urine = []
        clus_id = []
        if len(urine_mouse) > 0:
            m_urine_inds = urine_mouse == m
            m_urine = urine_data[m_urine_inds, :]
            # m_urine_seg = urine_seg[m_urine_inds]
            # _, clus_id = np.unique(m_urine_seg, return_inverse=True)
        out_dict = {'x_cm': mice_cents[:, 0, m],
                    'y_cm': mice_cents[:, 1, m],
                    'angle': angs[m],
                    'velocity': vels[m],
                    'ter_id': t_ids[m],
                    'urine_data': m_urine,
                    'urine_segment': clus_id,
                    'kpms_syllables': kpms_data,
                    'data_folder': folder_name}
        mouse_list.append(out_dict)

    if out_data is None and not skip_ptect:
        f_parts = os.path.split(folder_name)[-1]
        save_path = os.path.join(folder_name, f_parts + '_output.npy')
        np.save(save_path, mouse_list, allow_pickle=True)

    return mouse_list


def find_territory_files(root_dir: str):
    target_sufs = ['thermal.avi', 'thermal.h5', 'top.mp4', 'top.h5', 'ptmetadata.yml', 'fixedtop.h5', 'ptect.npz', 'output.npy']
    out_paths = {s: None for s in target_sufs}
    out_paths['root'] = root_dir
    for t in target_sufs:
        found_file = find_file_recur(root_dir, t)
        if found_file is not None:
            out_paths[t] = found_file
    return out_paths


def valid_dir(root_dir: str):
    target_sufs = ['thermal.avi', 'thermal.h5', 'top.mp4', 'fixedtop.h5']
    contains_f = np.zeros(len(target_sufs)).astype(bool)
    for ind, t in enumerate(target_sufs):
        has_t = find_file_recur(root_dir, t)
        contains_f[ind] = has_t is not None
    return np.all(contains_f)


def find_file_recur(root_fold, target_suf):
    files = os.listdir(root_fold)
    found_file = None
    for f in files:
        this_dir = os.path.join(root_fold, f)
        if os.path.isdir(this_dir):
            recur_has = find_file_recur(this_dir, target_suf)
            if recur_has is not None:
                found_file = recur_has
        elif f.split('_')[-1] == target_suf:
            found_file = this_dir
    return found_file


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


def clean_sleap_h5(slp_h5: str, block=1, orientation=0, cent_xy=(638, 504), suff='fixed'):
    rot_dict = {'rni': 0,
                'none': 0,
                'irn': 120,
                'nir': 240}
    rot_ang = orientation
    if type(orientation) is str:
        rot_ang = rot_dict[orientation]
    new_file = slp_h5.split('.')[0] + '_' + suff + '.h5'
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
            if i == 0 and np.all(np.isnan(last_cent)):
                first_cent_i = np.argwhere(~np.all(np.all(np.isnan(out_ts[0]), axis=0), axis=0))[0][0]
                last_cent = np.nanmean(out_ts[0, :, :, first_cent_i], axis=1)
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
            shutil.copy('../resources/nwb_metadata.yaml', yaml_file)
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


def make_anipose_directory(raw_data_folder, out_path):
    cam_dict = {'19060809': 'mid', '19194088': 'side', '19281943': 'top', '22049506': 'back'}
    targets = list(cam_dict.keys())
    sub_folds = os.listdir(raw_data_folder)
    for f in sub_folds:
        splitund = f.split('_')
        if splitund[-1] == 'thermal':
            print(f'os.mkdir(os.path.join({out_path}, {f}))')
        else:
            splitdot = f.split('.')
            if splitdot[-1] in targets:
                print(f'os.mkdir(os.path.join({out_path}, {f}))')


def load_kpms(kpms_csv, by_group=False):
    kpms_data = np.loadtxt(kpms_csv, delimiter=',', dtype=str)
    kpms_data = kpms_data[1:, 1:]
    names = np.unique(kpms_data[:, 0])
    kpms_dict = {}
    for n in names:
        this_data = kpms_data[kpms_data[:, 0] == n, :]
        frames = this_data[:, 7].astype(int)
        vec_len = max(frames) + 1
        syl_trace = np.zeros(vec_len)
        syl_trace[frames] = this_data[:, 6].astype(int)
        kpms_dict[n] = syl_trace
    return kpms_dict


if __name__ == '__main__':
    make_anipose_directory('Z:\\Dave\\LS Territory\\PPsync4\\runs\\PPsync4_Sub_DAB010_Stim_DAB015_RNI_Block0', 'Z:\\Dave\\LS Territory\\PPsync4\\runs\\PPsync4_Sub_DAB010_Stim_DAB015_RNI_Block0_ani')
