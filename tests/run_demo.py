import gdown
import os
from territorytools.gui import PtectGUI


def get_demo_data(google_drive_link, demo_fold='territorytools_demo'):
    abs_path = os.path.abspath(demo_fold)
    if not os.path.exists(abs_path):
        os.mkdir(abs_path)
        gdown.download_folder(google_drive_link)
    return abs_path


if __name__ == '__main__':
    fold_link = 'https://drive.google.com/drive/folders/1e58QlTkZTtZICjvpynQGK6FA5y0IkVGT?usp=sharing'
    demo_path = get_demo_data(fold_link)
    gui = PtectGUI(demo_path)