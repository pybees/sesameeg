# -*- coding: utf-8 -*-
""" Utility functions for the Sequential Monte Carlo algorithms."""
# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#          Alberto Sorrentino <sorrentino@dima.unige.it>
#
# License: BSD (3-clause)

import numpy as np
import scipy.special as sps
from mne.epochs import Epochs, EpochsArray
from mne.evoked import Evoked, EvokedArray
from mne.forward import Forward


def compute_neighbours_matrix(src, d_matrix, radius):
    """Compute the set of neighbours of each point in the brain discretization.

    Parameters
    -----------
    src :  :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts, 3)
        The coordinates of the points in the brain discretization.
    d_matrix : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts x n_verts)
        The Euclidean distance between the points in the
        brain discretization.
    radius : :py:class:`~float`
        The maximum distance between two neighbouring points.

    Returns
    --------
    n_matrix : :py:class:`~numpy.ndarray` of :py:class:`~int`, shape (n_verts, n_neigh_max)
        The sets of neighbours.
    """

    n_max = 100
    n_min = 3
    reached_points = np.array([0])
    counter = 0
    n_neigh = []
    list_neigh = []

    while (counter < reached_points.shape[0]
           and src.shape[0] > reached_points.shape[0]):
        P = reached_points[counter]
        aux = np.array(sorted(
            np.where(d_matrix[P] <= radius)[0],
            key=lambda k: d_matrix[P, k]))
        n_neigh.append(aux.shape[0])

        # Check the number of neighbours
        if n_neigh[-1] < n_min:
            raise ValueError('Computation of neighbours aborted since '
                             'their minimum number is too small.\n'
                             'Please choose a higher radius.')
        elif n_neigh[-1] > n_max:
            raise ValueError('Computation of neighbours aborted since'
                             'their maximum number is too big.\n'
                             'Please choose a lower radius.')
        list_neigh.append(aux)
        reached_points = np.append(reached_points,
                                   aux[~np.in1d(aux, reached_points)])
        counter += 1

    if counter >= reached_points.shape[0]:
        raise ValueError('Too small value of the radius:'
                         'the neighbour-matrix is not connected')
    elif src.shape[0] == reached_points.shape[0]:
        while counter < src.shape[0]:
            P = reached_points[counter]
            aux = np.array(sorted(
                np.where(d_matrix[P] <= radius)[0],
                key=lambda k: d_matrix[P, k]))
            n_neigh.append(aux.shape[0])

            if n_neigh[-1] < n_min:
                raise ValueError('Computation of neighbours aborted since '
                                 'their minimum number is too small.\n'
                                 'Please choose a higher radius.')
            elif n_neigh[-1] > n_max:
                raise ValueError('Computation of neighbours aborted since'
                                 'their maximum number is too big.\n'
                                 'Please choose a lower radius.')

            list_neigh.append(aux)
            counter += 1

        n_neigh_max = max(n_neigh)
        n_matrix = np.zeros([src.shape[0],
                          n_neigh_max], dtype=int) - 1
        for i in range(src.shape[0]):
            n_matrix[i, 0:list_neigh[i].shape[0]] = list_neigh[i]
        index_ord = np.argsort(n_matrix[:, 0])
        n_matrix = n_matrix[index_ord]
        return n_matrix

    else:
        raise RuntimeError("Some problems during"
                           "computation of neighbours.")


def compute_neighbours_probability_matrix(n_matrix, src, d_matrix, sigma_neigh):
    """Compute neighbours' probability matrix.

    Parameters
    -----------
    n_matrix : :py:class:`~numpy.ndarray` of :py:class:`~int`, shape (n_verts, n_neigh_max)
        The sets of neighbours.
    src :  :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts, 3)
        The coordinates of the points in the brain discretization.
    d_matrix : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts x n_verts)
        The Euclidean distance between the points in the
        brain discretization.
    sigma_neigh : :py:class:`~float`
        The standard deviation of the Gaussian distribution that defines
        the neighbours' probability.

    Returns
    --------
    np_matrix : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts, n_neigh_max)
        The neighbours' probability.
    """

    np_matrix = np.zeros(n_matrix.shape, dtype=float)
    for i in range(src.shape[0]):
        n_neig = len(np.where(n_matrix[i] > -1)[0])
        np_matrix[i, 0:n_neig] = \
            np.exp(-d_matrix[i, n_matrix[i, 0:n_neig]] ** 2
                   / (2 * sigma_neigh ** 2))
        np_matrix[i] = np_matrix[i] / np.sum(np_matrix[i])
    return np_matrix


def estimate_dip_mom_std(r_data, lf):
    """Estimate the standard deviation of the prior of the dipole moment.

    Parameters
    ----------
    r_data : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_sens, n_ist)
        The real part of the data; n_sens is the number of sensors and
        n_ist is the number of time-points or of frequencies.
    lf : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_sens x 3*n_verts)
        The leadfield matrix.

    Returns
    --------
    s_q : :py:class:`~float`
        The estimated standard deviation.
    """

    s_q = 1.5 * np.max(abs(r_data)) / np.max(abs(lf))
    return s_q


def estimate_noise_std(r_data):
    """Estimate the standard deviation of noise distribution.

    Parameters
    ----------
    r_data : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_sens, n_ist)
        The real part of the data; n_sens is the number of sensors and
        n_ist is the number of time-points or of frequencies.

    Returns
    --------
    s_noise : :py:class:`~float`
        The estimated standard deviation.
    """

    s_noise = 0.2 * np.max(abs(r_data))
    return s_noise


def gamma_pdf(x, shape, scale):
    """Evaluates the Gamma pdf on a single point.

    Parameters
    ----------
    x : :py:class:`~float`
        The point where to evaluate the Gamma pdf
    shape : :py:class:`~int`
        The Gamma pdf shape parameter :math:`k`
    scale : :py:class:`~float`
        The Gamma pdf scale parameter :math:`\\theta`

    Returns
    -------
    gamma_x : :py:class:`~float`
        The Gamma pdf evaluated on x
    """
    gamma_x = x**(shape-1)*(np.exp(-x/scale) / (sps.gamma(shape)*scale**shape))
    return gamma_x


def initialize_radius(src):
    """Guess the units of the points in the brain discretization and
    set to 1 cm the value of the radius for computing the sets of
    neighbours.

    Parameters
    -----------
    src :  :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts, 3)
        The coordinates of the points in the brain discretization.

    Returns
    --------
    radius : :py:class:`~float`
        The value of the radius.
    """

    x_length = (np.amax(src[:, 0]) - np.amin(src[:, 0]))
    y_length = (np.amax(src[:, 1]) - np.amin(src[:, 1]))
    z_length = (np.amax(src[:, 2]) - np.amin(src[:, 2]))

    max_length = max(x_length, y_length, z_length)

    if max_length > 50:
        radius = 10
    elif max_length > 1:
        radius = 1
    else:
        radius = 0.01

    return radius


def is_epochs(epch):
    if isinstance(epch, (Epochs, EpochsArray)):
        return True
    else:
        return False


def is_evoked(evk):
    if isinstance(evk, (Evoked, EvokedArray)):
        return True
    else:
        return False


def is_forward(fwd):
    if isinstance(fwd, Forward):
        return True
    else:
        return False


def sample_from_sphere():
    v = np.random.randint(0, 2)
    if v == 0:
        v = -1
    z = np.random.uniform(0, 1)
    phi = np.random.uniform(0, 2 * np.pi)

    theta = np.arccos(np.abs(z))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)

    out = np.array([x, y, z])
    out *= v
    out /= np.linalg.norm(out, axis=0)
    return out


def woodbury(A, U, V, k):
    A_inv_diag = 1./np.diag(A)
    B_inv = np.linalg.inv(np.eye(k) + (V * A_inv_diag) @ U)
    return np.diag(A_inv_diag) - (A_inv_diag.reshape(-1, 1) * U @ B_inv @ V * A_inv_diag)
