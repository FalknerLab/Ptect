import gdown
import os
import numpy as np
import matplotlib.pyplot as plt
from territorytools.gui import PtectApp
from territorytools.process import process_all_data


def get_demo_data(google_drive_link, demo_fold='territorytools_demo'):
    abs_path = os.path.abspath(demo_fold)
    if not os.path.exists(abs_path):
        os.mkdir(abs_path)
        gdown.download_folder(google_drive_link)
    return abs_path


if __name__ == '__main__':
    fold_link = 'https://drive.google.com/drive/folders/1e58QlTkZTtZICjvpynQGK6FA5y0IkVGT?usp=sharing'
    demo_path = get_demo_data(fold_link)
    # demo_path = 'D:\\ptect_dataset\\PPsync4_Resident_DAB014_Intruder_DAB019_Day_0_Orientation_IRN\\Block0'
    # gui = PtectApp(data_folder=demo_path)
    # out_data = np.load(os.path.join(demo_path, 'demo_ptect_full.npz'))
    # hd = out_data['hot_data']
    # plt.scatter(hd[:, 1], hd[:, 2], c=hd[:, 0])
    # plt.show()
    process_all_data(demo_path, skip_ptect=False, show_all=True)
