import numpy as np
from territorytools.urineV2 import proj_urine_across_time, urine_segmentation
import matplotlib.pyplot as plt
from territorytools.behavior import xy_to_polar
from scipy.stats import multivariate_normal


def make_synthetic_data(all_urine_xys):
    make_pdf(all_urine_xys)
    gen_urine_xys = np.array([0, 0])
    return gen_urine_xys


def make_pdf(all_urine_xy, bin_sz=100, data_ranges=((-np.pi, np.pi), (0, 32))):
    cdfs = []
    edges_list = []
    for dim, this_r in enumerate(data_ranges):
        this_pdf, bin_edges = np.histogram(all_urine_xy[:, dim], bins=bin_sz, density=True)
        norm_pdf = this_pdf / sum(this_pdf)
        cdf = np.cumsum(norm_pdf)
        cdfs.append(cdf)
        edges_list.append(bin_edges[1:])
        plt.plot(cdf)
    plt.show()
    cdf_array = np.array(cdfs).T
    edges_array = np.array(edges_list).T
    return edges_array, cdf_array


def sample_discrete_cdf(data_vals, cdfs, num_samps=1000):
    num_to_sample = cdfs.shape[1]
    rand_samps = np.random.uniform(0, 1, (num_samps, num_to_sample))
    samples = []
    for s in range(num_samps):
        samp_inds = np.argmax(cdfs >= rand_samps[s, :], axis=0)
        samples.append([data_vals[samp_inds[0], 0], data_vals[samp_inds[1], 1]])
    return np.array(samples)


if __name__ == '__main__':
    all_xys = np.ndarray((0, 2))
    for i in range(1):
        print('Loading ' + str(i))
        real_urine = np.load('D:/TerritoryTools/tests/test_data/urinexy' + str(i) + '.npy', allow_pickle=True)
        real_times = np.load('D:/TerritoryTools/tests/test_data/urinetimes' + str(i) + '.npy', allow_pickle=True)
        # urine_segmentation(real_times, real_urine)
        # proj_urine = proj_urine_across_time(real_urine, thresh=80)
        # all_xys = np.vstack((all_xys, proj_urine))
        urine_segmentation(real_times[:5000], real_urine[:5000])
    # all_t, all_r = xy_to_polar(all_xys)
    # x_vals, cdfs = make_pdf(np.vstack((all_t, all_r)).T)
    # samps = sample_discrete_cdf(x_vals, cdfs)
    # ax = plt.subplot(projection='polar')
    # ax.scatter(samps[:, 0], samps[:, 1])
    plt.show()
