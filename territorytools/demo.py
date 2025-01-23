import gdown
import os
from territorytools.gui import PtectApp


def get_demo_data(google_drive_link, demo_fold='tests/territorytools_demo'):
    """
    Downloads demo data from a Google Drive link and saves it to a specified folder.

    Parameters
    ----------
    google_drive_link : str
        URL to the Google Drive folder containing the demo data.
    demo_fold : str, optional
        Path to the folder where the demo data will be saved (default is 'tests/territorytools_demo').

    Returns
    -------
    str
        Absolute path to the folder containing the demo data.
    """
    abs_path = os.path.abspath(demo_fold)
    if not os.path.exists(abs_path):
        os.mkdir(abs_path)
        gdown.download_folder(google_drive_link)
    return abs_path

def run_demo():
    """
    Runs the demo by downloading the demo data and launching the PtectApp GUI.

    Returns
    -------
    None
    """
    fold_link = 'https://drive.google.com/drive/folders/1e58QlTkZTtZICjvpynQGK6FA5y0IkVGT?usp=sharing'
    demo_path = get_demo_data(fold_link)
    gui = PtectApp(data_folder=demo_path)

if __name__ == '__main__':
    run_demo()
