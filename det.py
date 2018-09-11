from matplotlib.mlab import normpdf
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import csv


def get_lbl_diff(diff_csv):
    file_arr = np.array(list(csv.reader(open(diff_csv))))
    labels = file_arr[:, 0].astype(np.int)
    diffs = file_arr[:, 1].astype(np.float)
    return labels, diffs


def get_det(labels, diffs):
    print('FNMR', sum(np.bitwise_and(labels == 1, diffs < 0)) / sum(labels == 1))
    print('FMR', sum(np.bitwise_and(labels == 0, diffs >= 0)) / sum(labels == 0))
    fnmr_list = list()
    fmr_list = list()
    for d in np.arange(diffs.min(), diffs.max(), step=0.01):
        fnmr_list.append(sum(np.bitwise_and(labels == 1, diffs < d)) / sum(labels == 1))
        fmr_list.append(sum(np.bitwise_and(labels == 0, diffs >= d)) / sum(labels == 0))
    return fnmr_list, fmr_list

    # bins = np.linspace(min(x + y), max(x + y), bin_num)
    bins = np.linspace(-10, +10, bin_num)
    plt.clf()
    plt.hist(x, bins, density=True, facecolor='g', alpha=0.5)
    plt.hist(y, bins, density=True, facecolor='r', alpha=0.5)
    plt.xlabel('Match_Score - Non_Match_Score')
    plt.title('Histogram of differences')
    plt.savefig('results/{}-hist.png'.format(tag))
    plt.close()


def plot_dstrb(labels, diffs):
    gen_diffs = diffs[labels == 1]
    imp_diffs = diffs[labels == 0]
    arange = np.arange(diffs.min(), diffs.max(), step=0.01)
    plt.plot(arange, normpdf(arange, np.mean(gen_diffs), np.std(gen_diffs)))
    plt.plot(arange, normpdf(arange, np.mean(imp_diffs), np.std(imp_diffs)))
    plt.show()


alex_lbl, alex_diffs = get_lbl_diff('results/24-alexnet-diff.csv')
plot_dstrb(alex_lbl, alex_diffs)
# alex_fnmr, alex_fmr = get_det(alex_lbl, alex_diffs)
# plt.plot(alex_fnmr, alex_fmr)
# plt.xlabel('FNMR')
# plt.ylabel('FMR')
# plt.show()
