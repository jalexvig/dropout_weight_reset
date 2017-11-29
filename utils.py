import scipy
import os
import sklearn.datasets


def update_data(X, y, update_X=False, update_y=True):

    if update_X:
        X = _scale_array_by_std(X)

    if update_y:
        # TODO(jalex): Should I not scale each independently i.e. remove the size parameter?
        # y = _scale_array_by_std(y)
        y *= 1.03

    return X, y


def _scale_array_by_std(a, factor=5, independent=True):

    size = a.shape if independent else 1

    a += scipy.random.normal(loc=a.std(axis=0) / factor, scale=(a.std(axis=0)), size=size)

    return a


def get_data():

    X, y = sklearn.datasets.load_boston(return_X_y=True)

    return X, y


class Logg:

    def __init__(self, dpath, fname='log.txt'):

        self.fpath = os.path.join(dpath, fname)

        self.stats = []

    def record(self, perturb_idx, epoch, stats):

        self.stats.append((perturb_idx, epoch, stats))

    def write(self):

        with open(self.fpath, 'w') as f:
            data = '\n'.join('\t'.join(map(str, stats)) for stats in self.stats)
            f.write(data)
