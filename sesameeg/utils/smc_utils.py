# -*- coding: utf-8 -*-
""" Utility functions for the Sequential Monte Carlo algorithms."""
# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#          Alberto Sorrentino <sorrentino@dima.unige.it>
#
# License: BSD (3-clause)

import numpy as np
import scipy.special as sps
import scipy.spatial.distance as ssd
from mne.epochs import Epochs, EpochsArray
from mne.evoked import Evoked, EvokedArray
from mne.forward import Forward
from mne import channel_type
from mne import pick_types_forward, read_labels_from_annot, add_source_space_distances


# def compute_cosine_distance(fwd):
#     """
#     Define cosine distance
#
#     Parameters:
#     -----------
#     fwd : instance of Forward
#         The forward solution
#
#     Returns:
#     --------
#
#     """
#
#     # Step 1. Understand what sensor type are present in the fwd solution
#     info = fwd['info']
#     picks = np.arange(0, info['nchan'])
#     types = np.array([channel_type(info, idx) for idx in picks])
#     ch_types_used = set(types)
#
#     # Step 2. Extract and normalized leadfield for each sensor type
#     aux_L = fwd['sol']['data']
#     L_norm = np.zeros(aux_L.shape)
#     for this_type in ch_types_used:
#         print('Normalizing leadfield for %s sensors' % this_type)
#         idx = picks[types == this_type]
#         Ns = idx.shape[0]
#         L_this_type = aux_L[idx]
#         L_this_type = L_this_type - np.mean(L_this_type, axis=0)
#         L_norm[idx] = L_this_type / \
#                       np.sqrt(1 / (Ns - 1) * np.sum(L_this_type ** 2, axis=0))
#
#     # Step 3. Compute cosine similarity matrix
#     cosine_distance = np.ones([L_norm.shape[1], L_norm.shape[1]]) - abs(np.corrcoef(L_norm.T))
#     return cosine_distance


def compute_correlation_distance_matix(fwd):
    print(' Computing correlation distance matrix...')
    distance_matrix = np.zeros((fwd['sol']['data'].shape[1], fwd['sol']['data'].shape[1]))
    n_ch_tot = fwd['info']['nchan']
    ch_types = set(map(lambda x: channel_type(fwd['info'], x), range(n_ch_tot)))

    for _t in ch_types:
        print('  Using {} sensors for computation...'.format(_t))
        if _t in ['mag', 'grad', 'planar1', 'planar2']:
            _fwd_t = pick_types_forward(fwd, meg=_t, ref_meg=False)
        elif _t == 'eeg':
            _fwd_t = pick_types_forward(fwd, eeg=_t, ref_meg=False)
        else:
            raise NotImplementedError

        n_ch_t = _fwd_t['info']['nchan']
        _lf_t = _fwd_t['sol']['data']
        _dm_t = ssd.cdist(_lf_t.T, _lf_t.T, metric='correlation')

        distance_matrix += (n_ch_t / n_ch_tot) * _dm_t

    print(' [done]')
    return distance_matrix


def compute_cortical_distance_matrix(fwd):
    src_aux = fwd['src'].copy()
    src_aux = add_source_space_distances(src_aux, dist_limit=np.inf, n_jobs=-1, verbose=True)
    print('Creating distance matrix...')
    # Left hemi
    lh_ixgrid = np.ix_(src_aux[0]['vertno'], src_aux[0]['vertno'])
    lh_m = src_aux[0]['dist'][lh_ixgrid].toarray()
    lh_shape = lh_m.shape[0]
    # Right hemi
    rh_ixgrid = np.ix_(src_aux[1]['vertno'], src_aux[1]['vertno'])
    rh_m = src_aux[1]['dist'][rh_ixgrid].toarray()
    rh_shape = rh_m.shape[0]
    # Create matrix
    distance_matrix = np.ones((fwd['nsource'], fwd['nsource'])) * 999
    distance_matrix[:lh_shape, :lh_shape] = lh_m
    distance_matrix[lh_shape:lh_shape + rh_shape, lh_shape:lh_shape + rh_shape] = rh_m
    print('    [done]')
    return distance_matrix


def compute_neighbours_matrix(src, d_matrix, radius, n_simil):
    """Compute the set of neighbours of each point in the brain discretization.

    Parameters
    ----------
    src :  :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts, 3)
        The coordinates of the points in the brain discretization.
    d_matrix : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts x n_verts)
        The Euclidean distance between the points in the
        brain discretization.
    radius : :py:class:`~float`
        The maximum distance between two neighbouring points.

    Returns
    -------
    n_matrix : :py:class:`~numpy.ndarray` of :py:class:`~int`, shape (n_verts, n_neigh_max)
        The sets of neighbours.
    """

    if n_simil == 1:
        return _compute_euclidean_neigh_matrix(src, d_matrix, radius)
    elif 0 <= n_simil < 1:
        return _compute_correlation_neigh_matrix(src, d_matrix, radius)
    else:
        raise NotImplementedError


def _compute_euclidean_neigh_matrix(src, d_matrix, radius):
    """Compute the set of neighbours of each point in the brain discretization.

    Parameters
    ----------
    src :  :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts, 3)
        The coordinates of the points in the brain discretization.
    d_matrix : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts x n_verts)
        The Euclidean distance between the points in the
        brain discretization.
    radius : :py:class:`~float`
        The maximum distance between two neighbouring points.

    Returns
    -------
    n_matrix : :py:class:`~numpy.ndarray` of :py:class:`~int`, shape (n_verts, n_neigh_max)
        The sets of neighbours.
    """

    n_max = 100
    n_min = 3
    reached_points = np.array([0])
    counter = 0
    n_neigh = []
    list_neigh = []

    while counter < reached_points.shape[0] < src.shape[0]:
        P = reached_points[counter]
        aux = np.array(sorted(
            np.where(d_matrix[P] <= radius)[0],
            key=lambda k: d_matrix[P, k]))
        n_neigh.append(aux.shape[0])

        # Check the number of neighbours
        if n_neigh[-1] < n_min:
            raise ValueError('Computation of neighbours aborted since '
                             'their minimum number is too small.\n'
                             f'Please choose a radius higher than {radius}.')
        elif n_neigh[-1] > n_max:
            raise ValueError('Computation of neighbours aborted since'
                             'their maximum number is too big.\n'
                             f'Please choose a radius lower than {radius}.')
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


def _compute_correlation_neigh_matrix(src, d_matrix, n_neigh_init):
    n_vert = src.shape[0]

    # For each vertex find the set of k-nearest neighbours
    set_n_neigh = np.argsort(d_matrix, axis=1)[:, :n_neigh_init]

    # Fill in the set of nearest neighbour so to have a reversible kernel
    adjacency_matrix = np.zeros(d_matrix.shape)
    for ir in np.arange(n_vert):  # Avoid this for?!?!?!
        adjacency_matrix[ir, set_n_neigh[ir]] = 1
    adjacency_matrix = ((adjacency_matrix.T + adjacency_matrix) > 0)
    n_neigh = np.sum(adjacency_matrix, axis=1)

    # Reorder the set of neighbours and check if the obtained matrix is connected
    n_matrix = np.zeros([n_vert, np.max(n_neigh)], dtype=int) - 1
    counter = 0
    reached_points = np.array([0])

    while counter < reached_points.shape[0]:
        P = reached_points[counter]
        aux = np.array(sorted(
            np.where(adjacency_matrix[P])[0],
            key=lambda k: d_matrix[P, k]))
        n_matrix[P, :aux.shape[0]] = aux
        reached_points = np.append(reached_points,
                                   aux[~np.in1d(aux, reached_points)])
        counter += 1
    if reached_points.shape[0] < n_vert:
        raise ValueError('Too small value for the initial number of nearest neighbour:'
                         'the neighbour-matrix is not connected')
    else:
        return n_matrix


def compute_neighbours_probability_matrix(n_matrix, src, d_matrix, sigma_neigh):
    """Compute neighbours' probability matrix.

    Parameters
    ----------
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
    -------
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
    lf : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_sens x n_comp*n_verts)
        The leadfield matrix. (n_comp = 1, if fixed orientation, 3, if free orientation)

    Returns
    -------
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
    -------
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
    ----------
    src :  :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts, 3)
        The coordinates of the points in the brain discretization.

    Returns
    -------
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


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def prior_loc_from_labels(subject, subjects_dir, fwd, parc, sel_labels, ratio=None):
    """Construct the prior probability of active source locations starting from given FreeSurfer/MNE labels.

    Parameters
    ----------
    subject : :py:class:`~str`
        Subject name in Freesurfer subjects dir. Only when instantiated by means
        of :func:`~sesameeg.mne.prepare_sesame`.
    subjects_dir : :py:class:`~str` | None
        If not None, this directory will be used as the subjects directory instead of the value set using
        the SUBJECTS_DIR environment variable. Only when instantiated by means
        of :func:`~sesameeg.mne.prepare_sesame`
    fwd : :py:class:`~mne.Forward` object
        The forward solution.
    parc : :py:class:`~str`
        The parcellation to use, e.g., ``'aparc'`` or ``'aparc.a2009s'``.
    sel_labels : :py:class:`~list` of :py:class:`~str`
        The cortical labels to use.
    ratio : :py:class:`~float`
        The ratio of high prior probability cortical areas to low prior probability cortical areas
        probability value. If None, it is set to 1.

    Returns
    -------
    prior_loc_arr : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts, )
        The prior probability of active source locations.
    """

    labels = read_labels_from_annot(subject=subject, parc=parc, hemi='both',
                                    surf_name='inflated', subjects_dir=subjects_dir)

    def get_idx(sel_label):
        for i, l in enumerate(labels):
            if l.name == sel_label:
                return i

    _ssgrid = fwd['source_rr']
    _src_vertno_lh = fwd['src'][0]['vertno']
    _src_vertno_rh = fwd['src'][1]['vertno']
    if ratio is None:
        prior_loc_arr = np.zeros(_ssgrid.shape[0])
    else:
        prior_loc_arr = 1e-6 * np.ones(_ssgrid.shape[0])
    labels_idx = list(map(lambda x: get_idx(x), sel_labels))
    for _l_idx in labels_idx:
        _l = labels[_l_idx]
        if _l.name.endswith('lh'):
            vertices0 = _l.get_vertices_used(_src_vertno_lh)
            vertices1 = [_v for _v in vertices0 if _v in _src_vertno_lh]
            vertices2 = np.asarray(list(map(lambda x: np.argwhere(_src_vertno_lh == x)[0][0], vertices1)))
            if ratio is None:
                prior_loc_arr[vertices2] += 1
            else:
                prior_loc_arr[vertices2] *= ratio
        elif _l.name.endswith('rh'):
            vertices0 = _l.get_vertices_used(_src_vertno_rh)
            vertices1 = [_v for _v in vertices0 if _v in _src_vertno_rh]
            vertices2 = np.asarray(list(map(lambda x: np.argwhere(_src_vertno_rh == x)[0][0], vertices1)))
            if ratio is None:
                prior_loc_arr[vertices2 + _src_vertno_lh.shape[0]] += 1
            else:
                prior_loc_arr[vertices2 + _src_vertno_lh.shape[0]] *= ratio
        else:
            raise ValueError
    return prior_loc_arr


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
