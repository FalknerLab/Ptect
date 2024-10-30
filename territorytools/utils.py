import os


def find_rename_cam_videos(root_dir, cam_dict=None):
    if cam_dict is None:
        cam_dict = {'19060809': 'side1',
                    '19194088': 'side3',
                    '19281943': 'top',
                    '22049506': 'side2'}
    conts = os.listdir(root_dir)
    for f in conts:
        f_suf = f.split('.')
        next_dir = os.path.join(root_dir, f)
        if f_suf[-1] in cam_dict.keys():
            vid_file = os.path.join(next_dir, '000000.mp4')
            daq_file = os.path.join(next_dir, '000000.npz')
            md_file = os.path.join(next_dir, 'metadata.yaml')
            new_vid = os.path.join(next_dir, f'{f_suf[0]}_{cam_dict[f_suf[-1]]}.mp4')
            new_daq = os.path.join(next_dir, f'{f_suf[0]}_{cam_dict[f_suf[-1]]}.npz')
            new_md = os.path.join(next_dir, f'{f_suf[0]}_{cam_dict[f_suf[-1]]}_metadata.yaml')
            for s, d in zip((vid_file, daq_file, md_file), (new_vid, new_daq, new_md)):
                os.rename(s, d)
        if os.path.isdir(next_dir):
            find_rename_cam_videos(next_dir, cam_dict=cam_dict)


if __name__ == '__main__':
    root_dir = 'Z:/Dave/LS Territory/PPsync4/runs/'
    find_rename_cam_videos(root_dir)
