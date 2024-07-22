import os
import h5py
import numpy as np
from process import valid_dir, import_all_data
from ttclasses import BasicExp, BasicRun
from modeling import fit_glm
from behavior import make_design_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def load_dataset(data_dir):
    runs = []
    for f in os.listdir(data_dir):
        this_path = os.path.join(data_dir, f)
        if valid_dir(this_path):
            print(this_path)
            run_data, run_info = import_all_data(this_path, show_all=False)
            this_run = BasicRun(run_data, run_info['Territory'])
            runs.append(this_run)
    return BasicExp(runs)


def run_regression_formation(exp_data: BasicExp):
    form_data = exp_data.filter_by_group('block', '0')
    all_data = []
    for r in form_data.runs:
        try:
            design = make_design_matrix(r.data, r.info)
            all_data.append(design)
        except IndexError:
            z = 0
    all_data = np.hstack(all_data).T
    urine_inds = [4, 9]
    # urine_inds = np.arange(0, 12)
    for u in urine_inds:
        this_des = np.delete(all_data, u, 1)
        # this_des = all_data
        urine_target = all_data[:, u]
        y_test, pred, full_pred, r2, model = fit_glm(this_des, urine_target)
        print(r2)
        plt.plot(urine_target)
        plt.plot(full_pred)
        plt.show()


if __name__ == '__main__':
    data_dir = 'D:/TerritoryTools/data'
    # make_save_design_matrix(data_dir)
    full_exp = load_dataset(data_dir)
    run_regression_formation(full_exp)
