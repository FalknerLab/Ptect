import os
import h5py
import numpy as np


def find_rename_cam_videos(root_dir, cam_dict=None):
    """
    Finds and renames camera video files based on a given dictionary.

    Parameters
    ----------
    root_dir : str
        Root directory to search.
    cam_dict : dict, optional
        Dictionary mapping folder suffixes to new names (default is None).

    Returns
    -------
    None
    """
    if cam_dict is None:
        cam_dict = {'19060809': 'side1',
                    '19194088': 'side3',
                    '19281943': 'top',
                    '22049506': 'side2'}
    conts = os.listdir(root_dir)
    for f in conts:
        f_suf = f.split('.')
        root_n = os.path.split(root_dir)[-1]
        next_dir = os.path.join(root_dir, f)
        if f_suf[-1] in cam_dict.keys():
            vid_file = os.path.join(next_dir, '000000.mp4')
            daq_file = os.path.join(next_dir, '000000.npz')
            md_file = os.path.join(next_dir, 'metadata.yaml')
            new_vid = os.path.join(next_dir, f'{root_n}_{cam_dict[f_suf[-1]]}.mp4')
            new_daq = os.path.join(next_dir, f'{root_n}_{cam_dict[f_suf[-1]]}.npz')
            new_md = os.path.join(next_dir, f'{root_n}_{cam_dict[f_suf[-1]]}_metadata.yaml')
            if os.path.exists(vid_file):
                for s, d in zip((vid_file, daq_file, md_file), (new_vid, new_daq, new_md)):
                    os.rename(s, d)
        if os.path.isdir(next_dir):
            find_rename_cam_videos(next_dir, cam_dict=cam_dict)


def fix_sleap_h5(slp_h5: str, block=1, orientation=0, cent_xy=(638, 504), suff='fixed', dist_thresh=25, chunk_sz=16384):
    """
    Fixes the SLEAP H5 file by updating tracking scores, instance scores, point scores, track occupancy, and tracks.

    Parameters
    ----------
    slp_h5 : str
        Path to the SLEAP H5 file.
    block : int, optional
        Block number to process (default is 1).
    orientation : int or str, optional
        Orientation of the data (default is 0).
    cent_xy : tuple, optional
        Center coordinates (x, y) (default is (638, 504)).
    suff : str, optional
        Suffix for the new file name (default is 'fixed').
    dist_thresh : int, optional
        Distance threshold for filtering (default is 25).
    chunk_sz : int, optional
        Chunk size for processing (default is 16384).

    Returns
    -------
    str
        Path to the new fixed H5 file.
    """
    rot_dict = {'rni': 0,
                'none': 0,
                'irn': 120,
                'nir': 240}
    rot_ang = orientation
    if type(orientation) is str:
        rot_ang = rot_dict[orientation]
    new_file = slp_h5.split('.')[0] + '_' + suff + '.h5'
    slp_data = h5py.File(slp_h5, 'r')
    fixed_file = h5py.File(new_file, 'a')
    len_t = slp_data['tracks'].shape[3]
    node_num = slp_data['tracks'].shape[2]
    out_name = ''
    out_t_scores = np.zeros((1, len_t))
    out_i_scores = np.zeros((1, len_t))
    out_p_scores = np.zeros((1, node_num, len_t))
    out_t_occs = np.zeros((len_t, 1))
    out_ts = None
    num_chks = int(np.ceil(len_t/chunk_sz))
    if block:
        out_name = slp_data['track_names'][0][:]
        out_ts = slp_data['tracks'][0][None, :]
        last_cent = np.nanmean(out_ts[0, :, :, 0], axis=1)
        for c in range(num_chks):
            print('Cleaning slp file, on frame: ', c*chunk_sz)
            inds = np.arange(c*chunk_sz, (c+1)*chunk_sz)
            inds = inds[inds < len_t]
            tracks = slp_data['tracks'][:, :, :, inds]
            t_scores = slp_data['tracking_scores'][:, inds]
            i_scores = slp_data['instance_scores'][:, inds]
            p_scores = slp_data['point_scores'][:, :, inds]
            t_occupancy = slp_data['track_occupancy'][inds, :]
            for i in range(len(inds)):
                if i == 0 and c == 0 and np.all(np.isnan(last_cent)):
                    first_cent_i = np.argwhere(~np.all(np.all(np.isnan(out_ts[0]), axis=0), axis=0))[0][0]
                    last_cent = np.nanmean(out_ts[0, :, :, first_cent_i], axis=1)

                this_cent = np.nanmean(tracks[:, :, :, i], axis=2)
                dists = np.linalg.norm(this_cent - last_cent, axis=1)
                if not np.all(np.isnan(dists)):
                    in_thresh = dists < dist_thresh
                    if np.any(in_thresh):
                        i_scores[~in_thresh, i] = -np.inf
                    best_track = np.nanargmax(i_scores[:, i])
                    out_t_scores[:, inds[i]] = t_scores[best_track, i]
                    out_p_scores[0, :, inds[i]] = p_scores[best_track, :, i]
                    out_i_scores[0, inds[i]] = i_scores[best_track, i]
                    last_cent = this_cent[best_track, :]
                    out_t_occs[inds[i], 0] = t_occupancy[i, best_track]
                    out_ts[0, :, :, inds[i]] = tracks[best_track, :, :, i]
        fixed_file['track_names'] = [out_name]

    else:
        out_names = slp_data['track_names'][:2][:]
        tracks = np.array(slp_data['tracks'])
        t_scores = np.array(slp_data['tracking_scores'])
        i_scores = np.array(slp_data['instance_scores'])
        p_scores = np.array(slp_data['point_scores'])
        t_occupancy = np.array(slp_data['track_occupancy'])
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
        out_t_scores = t_scores[:2, :]
        out_i_scores = i_scores[:2, :]
        out_p_scores = p_scores[:2, :, :]
        out_t_occs = t_occupancy[:, :2]
        fixed_file['track_names'] = out_names
    fixed_file['tracking_scores'] = out_t_scores
    fixed_file['instance_scores'] = out_i_scores
    fixed_file['point_scores'] = out_p_scores
    fixed_file['track_occupancy'] = out_t_occs
    fixed_file['tracks'] = out_ts
    fixed_file['edge_inds'] = slp_data['edge_inds'][:]
    fixed_file['edge_names'] = slp_data['edge_names'][:]
    fixed_file['labels_path'] = ()
    fixed_file['node_names'] = slp_data['node_names'][:]
    fixed_file['provenance'] = ()
    fixed_file['video_ind'] = ()
    fixed_file['video_path'] = ()
    return new_file


def xy_to_cm(xy, center_pt=(325, 210), px_per_cm=225/30.48):
    """
    Converts pixel coordinates to centimeters.

    Parameters
    ----------
    xy : numpy.ndarray
        Array of pixel coordinates.
    center_pt : tuple, optional
        Center point for conversion (default is (325, 210)).
    px_per_cm : float, optional
        Pixels per centimeter (default is 225/30.48).

    Returns
    -------
    cm_x : numpy.ndarray
        X coordinates in centimeters.
    cm_y : numpy.ndarray
        Y coordinates in centimeters.
    """
    rel_xy = xy - center_pt
    rel_xy[:, 1] = -rel_xy[:, 1]
    cm_x = rel_xy[:, 0] / px_per_cm
    cm_y = rel_xy[:, 1] / px_per_cm
    return cm_x, cm_y


def xy_to_cm_vec(xys, center_pt=(325, 210), px_per_cm=225/30.48):
    """
    Converts a vector of pixel coordinates to centimeters.

    Parameters
    ----------
    xys : numpy.ndarray
        Array of pixel coordinates.
    center_pt : tuple, optional
        Center point for conversion (default is (325, 210)).
    px_per_cm : float, optional
        Pixels per centimeter (default is 225/30.48).

    Returns
    -------
    numpy.ndarray
        Array of coordinates in centimeters.
    """
    xys_rot = []
    for i in range(xys.shape[-1]):
        this_xy = xys[:, :, i].T
        cm_x, cm_y = xy_to_cm(this_xy, center_pt=center_pt, px_per_cm=px_per_cm)
        xys_rot.append(np.vstack((cm_x, cm_y)))
    xys_rot = np.array(xys_rot)
    return np.moveaxis(xys_rot, 0, 2)


def rotate_xy_vec(pts, rot_ang):
    """
    Rotates a vector of coordinates by a given angle.

    Parameters
    ----------
    pts : numpy.ndarray
        Array of coordinates to rotate.
    rot_ang : float
        Rotation angle in degrees.

    Returns
    -------
    numpy.ndarray
        Array of rotated coordinates.
    """
    out_pts = []
    for i in range(pts.shape[-1]):
        rot_pts = rotate_xy(pts[0, :, i], pts[1, :, i], rot_ang)
        out_pts.append(np.vstack(rot_pts))
    xys_rot = np.array(out_pts)
    return np.moveaxis(xys_rot, 0, 2)


def rotate_xy(x, y, rot):
    """
    Rotates coordinates by a given angle.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates.
    y : numpy.ndarray
        Y coordinates.
    rot : float or str
        Rotation angle in degrees or a string key for predefined angles.

    Returns
    -------
    tuple
        Rotated X and Y coordinates.
    """
    rot_dict = {'rni': 0,
                'none': 0,
                'irn': 120,
                'nir': 240,
                'RNI': 0,
                'IRN': 120,
                'NIR': 240}
    if type(rot) == str:
        rot_deg = rot_dict[rot]
    else:
        rot_deg = rot
    in_rad = np.radians(rot_deg)
    c, s = np.cos(in_rad), np.sin(in_rad)
    rot_mat = [[c, -s], [s, c]]
    xy = rot_mat @ np.vstack((x, y))
    return xy[0, :], xy[1, :]


def intersect2d(arr0, arr1, return_int=True):
    """
    Finds the intersection of two 2D arrays.

    Parameters
    ----------
    arr0 : numpy.ndarray
        First array.
    arr1 : numpy.ndarray
        Second array.
    return_int : bool, optional
        Whether to return the intersection (default is True).

    Returns
    -------
    numpy.ndarray or tuple
        Intersection of the arrays or the non-intersecting elements and a boolean array.
    """
    cool_evts_1d = list([''.join(str(row)) for row in arr0])
    te_1d = list([''.join(str(row)) for row in arr1])
    valid_int, te_inds, ce_inds = np.intersect1d(te_1d, cool_evts_1d, return_indices=True)
    if return_int:
        return arr0[ce_inds, :]
    else:
        bool_arr = np.ones_like(arr1[:, 0]).astype(bool)
        bool_arr[te_inds] = False
        return arr1[bool_arr, :], bool_arr


def find_rename_cam_folders(root_dir, cam_dict=None):
    """
    Finds and renames camera folders based on a given dictionary.

    Parameters
    ----------
    root_dir : str
        Root directory to search.
    cam_dict : dict, optional
        Dictionary mapping folder suffixes to new names (default is None).

    Returns
    -------
    None
    """
    if cam_dict is None:
        cam_dict = {'19060809': 'side1',
                    '19194088': 'side3',
                    '19281943': 'top',
                    '22049506': 'side2'}
    conts = os.listdir(root_dir)
    for f in conts:
        next_dir = os.path.join(root_dir, f)
        t_split = f.split('_')[-1]
        if t_split == 'thermal':
            os.rename(next_dir, os.path.join(root_dir, 'thermal'))
        else:
            f_suf = f.split('.')
            if f_suf[-1] in cam_dict.keys():
                os.rename(next_dir, os.path.join(root_dir, cam_dict[f_suf[-1]]))
        if os.path.isdir(next_dir):
            find_rename_cam_folders(next_dir, cam_dict=cam_dict)

def rename_ri(root_dir):
    for root in os.listdir(root_dir):
        this_dir = os.path.join(root_dir, root)
        new_name = this_dir.replace('resident', 'self')
        new_name = new_name.replace('intruder', 'other')
        os.rename(this_dir, new_name)

    for root, dirs, files in os.walk(root_dir, topdown=False):
        for f in files:
            this_dir = os.path.join(root, f)
            new_dir = this_dir.lower()
            new_dir = new_dir.replace('resident', 'self')
            new_dir = new_dir.replace('intruder', 'other')
            os.rename(this_dir, new_dir)


if __name__ == '__main__':
    # root_dir = 'D:/ptect_dataset/testing/kpms'
    # for f in os.listdir(root_dir):
    #     file_n = os.path.join(root_dir, f)
    fix_sleap_h5('D:\\ptect_dataset\\AggExp\\data\\'
                 'aggexppilot_self_dab019_other_dab013_day_1_block_0_orientation_rni_20250127_125723\\'
                 'aggexppilot_self_dab019_other_dab013_day_1_block_0_orientation_rni_20250127_125723_top_old.h5',
                 block=0, orientation='rni', cent_xy=(707, 541))

    # root_dir = 'D:/ptect_dataset/kpms/grid_movies'
    # for f in os.listdir(root_dir):
    #     f_path = os.path.join(root_dir, f)
    #     f_split = f.split('.')
    #     if f_split[-1] == 'mp4':
    #         out_f = os.path.join(root_dir, ''.join(f_split[:-1]) + '.gif')
    #         command = f'ffmpeg -i {f_path} {out_f}'
    #         print(command)
    #         os.system(command)

    # rename_ri('D:\\ptect_dataset\\AggExp\\data')
    # find_rename_cam_videos('D:\\ptect_dataset\\AggExp\\data', cam_dict={'top':'top', 'side1':'side1', 'side2':'side2', 'side3':'side3'})