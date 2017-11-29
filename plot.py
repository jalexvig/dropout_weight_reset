import re
from matplotlib import pyplot as plt
import seaborn as sns
import os
import pandas as pd


def plot(dpath_root, save=False):

    for dpath, dnames, fnames in os.walk(dpath_root):

        if 'log.txt' not in fnames:
            continue

        fpath = os.path.join(dpath, 'log.txt')

        df = pd.read_csv(fpath, sep='\t', index_col=[0, 1], header=None)
        df.index.names = ['Data Idx', 'Epoch']
        df = df.iloc[:, 0].unstack(level=0)

        fname = os.path.split(dpath)[-1]
        name = re.match('(.*)_\d{4}-\d{2}-\d{2}', fname).group(1)

        df.plot(title=name)

        plt.xlim(0, 400)

        plt.ylim(40, 150)
        plt.ylabel('Loss')

        if save:
            fpath_save = os.path.join(dpath_root, 'losses_%s.png' % name)
            plt.gcf().set_size_inches(10, 8)
            plt.savefig(fpath_save)
        else:
            plt.show()


# def _plot(dpaths, save):
#
#     for dpath in dpaths:
#
#         fpath = os.path.join(dpath, 'log.txt')
#
#         df = pd.read_csv(fpath, sep='\t', index_col=[0, 1], header=None)
#         df.index.names = ['Data Idx', 'Epoch']
#         df = df.iloc[:, 0].unstack(level=0)
#
#         fname = os.path.split(dpath)[-1]
#         name = re.match('(.*)_\d{4}-\d{2}-\d{2}', fname).group(1)
#
#         df.plot(title=name)
#
#         plt.xlim(0, 400)
#
#         plt.ylim(40, 150)
#         plt.ylabel('Loss')
#
#         if save:
#             fpath_save = os.path.join(dpath, 'losses.png')
#             plt.gcf().set_size_inches(10, 8)
#             plt.savefig(fpath_save)
#         else:
#             plt.show()


if __name__ == '__main__':

    plot('/Users/alex/ml/dropout_weight_reset/data', save=True)
