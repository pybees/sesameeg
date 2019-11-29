# -*- coding: utf-8 -*-

# Author: Gianvittorio Luria <luria@dima.unige.it>
#
# License: BSD (3-clause)

import numpy as np
import itertools as it
import scipy.spatial.distance as ssd
from ..utils import initialize_radius


def compute_mld(src, blob, true_locs):
    """
    :param src:
    :param blob:
    :param true_locs:
    :return:
        mld in millimetri
    """
    if blob.shape[0] != true_locs.shape[0]:
        raise ValueError
    mld = list()
    _r = initialize_radius(src)
    if _r == 0.01:
        _scale = 1000
    elif _r == 1:
        _scale= 10
    elif _r == 10:
        _scale = 1
    else:
        raise ValueError

    for _tl, _b in zip(true_locs, blob):
        _true_coord = src[_tl]
        _dist = list(map(lambda x: _scale * ssd.euclidean(x, _true_coord), src))
        _num = np.sum(list(map(lambda x, y: (x * np.abs(y))**2, _dist, _b)))
        _denum = np.sum(np.abs(_b)**2)
        mld.append(np.sqrt(_num / _denum))
    return mld


def compute_ospa(src, true_locs, est_locs, true_src=None):
    """

    :param src: array
            Source space
    :param true_locs: array
    :param est_locs: array
    :return:
    """

    if true_src is not None:
        distance_matrix = ssd.cdist(true_src, src)
    else:
        distance_matrix = ssd.cdist(src, src)
    true_n_dips = true_locs.shape[0]
    est_n_dips = est_locs.shape[0]

    if est_n_dips <= true_n_dips:
        all_perms_idx = np.asarray(list(it.permutations(range(true_n_dips))))
        ospa_temp = np.mean(distance_matrix[true_locs[all_perms_idx][:, :est_n_dips], est_locs],
                            axis=1)
        ospa = np.min(ospa_temp)
    elif est_n_dips > true_n_dips:
        all_perms_idx = np.asarray(list(it.permutations(range(est_n_dips))))
        ospa_temp = np.mean(distance_matrix[true_locs, est_locs[all_perms_idx][:, :true_n_dips]],
                            axis=1)
        ospa = np.min(ospa_temp)
    else:
        raise ValueError

    return ospa


def compute_sd(src, blob):
    """
    :param src:
    :param blob:
    :return:
       sd in millimetri
    """
    sd = list()
    _r = initialize_radius(src)
    if _r == 0.01:
        _scale = 1000
    elif _r == 1:
        _scale= 10
    elif _r == 10:
        _scale = 1
    else:
        raise ValueError

    for _b in blob:
        _peak = np.argmax(_b)
        _peak_coord = src[_peak]
        _dist = list(map(lambda x: _scale * ssd.euclidean(x, _peak_coord), src))
        _num = np.sum(list(map(lambda x, y: (x * np.abs(y))**2, _dist, _b)))
        _denum = np.sum(np.abs(_b)**2)
        sd.append(np.sqrt(_num / _denum))
    return sd
