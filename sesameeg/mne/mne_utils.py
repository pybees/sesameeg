import numpy as np
import scipy.spatial.distance as ssd
from mne.cov import compute_whitener
from mne.epochs import Epochs, EpochsArray
from mne.evoked import Evoked, EvokedArray
from mne.forward import Forward
from mne.forward.forward import _select_orient_forward
from mne import channel_type
from mne import pick_types_forward, add_source_space_distances
from ..utils import normalize, initialize_radius, compute_neighbours_matrix, compute_neighbours_probability_matrix
from ..sesame import Sesame


def _is_epochs(data):
    if isinstance(data, (Epochs, EpochsArray)):
        return True
    else:
        return False


def _is_evoked(data):
    if isinstance(data, (Evoked, EvokedArray)):
        return True
    else:
        return False


def _is_forward(data):
    if isinstance(data, Forward):
        return True
    else:
        return False


def _get_topographies(evoked, top_min, top_max, fourier=False, freqs=None):
    if fourier is False:
        if top_min is None:
            s_min = 0
            top_min = evoked.times[0]
        else:
            if isinstance(top_min, float):
                s_min = evoked.time_as_index(top_min, use_rounding=True)[0]
                top_min = evoked.times[s_min]
            else:
                raise ValueError('top_min value should be a float')

        if top_max is None:
            s_max = evoked.data.shape[1] - 1
            top_max = evoked.times[-1]
        else:
            if isinstance(top_max, float):
                s_max = evoked.time_as_index(top_max, use_rounding=True)[0]
                top_max = evoked.times[s_max]
            else:
                raise ValueError('top_max value should be a float')
        print(f'Analyzing data from {round(top_min, 4)} s to {round(top_max, 4)} s')
        return s_min, s_max, top_min, top_max
    elif fourier is True:
        if top_min is None:
            s_min = 0
            top_min = freqs[0]
        else:
            if isinstance(top_min, float):
                s_min = np.where((freqs >= top_min))[0][0]
                top_min = freqs[s_min]
            else:
                raise ValueError('top_min value should be a float')

        if top_max is None:
            s_max = evoked.data.shape[1] - 1
            top_max = freqs[-1]
        else:
            if isinstance(top_max, float):
                s_max = np.where((freqs <= top_max))[0][-1]
                top_max = freqs[s_max]
            else:
                raise ValueError('top_max value should be a float')
        print(f'Analyzing data from {round(top_min, 4)} Hz to {round(top_max, 4)} Hz')
        return s_min, s_max, top_min, top_max
    else:
        raise ValueError


def _prepare_epochs_data(epochs, top_min, top_max, epochs_avg=False, subsample=None, fourier=False):
    ep_data = epochs.get_data()
    evoked = EvokedArray(ep_data[0], epochs.info, tmin=epochs.times[0])
    if fourier is False:
        s_min, s_max, top_min, top_max = _get_topographies(evoked, top_min, top_max, fourier=fourier)
        _times = evoked.times[s_min:s_max + 1]
        _sfreq = evoked.info['sfreq']
        temp_list = list()
        for ie, _e in enumerate(ep_data):
            if subsample is not None:
                if ie == 0:
                    print(f'Subsampling data with step {subsample}')
                temp_list.append(_e[:, s_min:s_max + 1:subsample])
            else:
                temp_list.append(_e[:, s_min:s_max + 1])
        if epochs_avg is True:
            print(f'Averaging {len(temp_list)} epochs.')
            return np.mean(np.array(temp_list), axis=0), _times
        else:
            return np.hstack(temp_list), _times, _sfreq
    elif fourier is True:
        _sfreq = evoked.info['sfreq']
        tstep = 1 / evoked.info['sfreq']
        evoked_f = evoked.copy()
        evoked_f.data *= np.hamming(evoked.data.shape[1])
        evoked_f.data = (np.fft.rfft(evoked_f.data))
        freqs = np.fft.rfftfreq(evoked.data.shape[1], tstep)
        print('Data have been converted to the frequency domain.')

        s_min, s_max, top_min, top_max = _get_topographies(evoked_f, top_min, top_max, fourier=fourier, freqs=freqs)
        _freqs = freqs[s_min:s_max + 1]

        temp_list = list()
        temp_list2 = list()
        temp_list2_r = list()
        temp_list2_i = list()
        for ie, _e in enumerate(ep_data):
            _e *= np.hamming(_e.shape[1])
            _e_f = np.fft.rfft(_e)
            if subsample is not None:
                if ie == 0:
                    print('Subsampling data with step {0}'.format(subsample))
                temp_list.append(_e_f[:, s_min:s_max + 1:subsample])
            else:
                temp_list.append(_e_f[:, s_min:s_max + 1])
           # self._tl = temp_list ATTENZIONE!!! dove lo uso?

        for _data_temp in temp_list:
            temp_list2.append(np.real(_data_temp))
            temp_list2.append(np.imag(_data_temp))

            temp_list2_r.append(np.real(_data_temp))
            temp_list2_i.append(np.imag(_data_temp))

        if epochs_avg is True:
            print(f'Averaging {len(temp_list)} epochs.')
            _r_mean = np.mean(np.array(temp_list2_r), axis=0)
            _i_mean = np.mean(np.array(temp_list2_i), axis=0)
            return np.hstack([_r_mean, _i_mean]), _freqs, _sfreq
        else:
            return np.hstack(temp_list2), _freqs, _sfreq
    else:
        raise ValueError


def _prepare_evoked_data(evoked, top_min, top_max, subsample=None, fourier=False):
    if fourier is False:
        s_min, s_max, top_min, top_max = _get_topographies(evoked, top_min, top_max, fourier=fourier)
        _times = evoked.times[s_min:s_max + 1]
        _sfreq = evoked.info['sfreq']

        if subsample is not None:
            print(f'Subsampling data with step {subsample}')
            _data = evoked.data[:, s_min:s_max + 1:subsample]
        else:
            _data = evoked.data[:, s_min:s_max + 1]
        return _data, _times, _sfreq
    elif fourier is True:
        _sfreq = evoked.info['sfreq']
        tstep = 1 / evoked.info['sfreq']
        evoked_f = evoked.copy()
        evoked_f.data *= np.hamming(evoked.data.shape[1])
        evoked_f.data = (np.fft.rfft(evoked_f.data))
        freqs = np.fft.rfftfreq(evoked.data.shape[1], tstep)
        print('Data have been converted to the frequency domain.')

        s_min, s_max, top_min, top_max = _get_topographies(evoked_f, top_min, top_max, freqs=freqs, fourier=fourier)
        _freqs = freqs[s_min:s_max + 1]

        if subsample is not None:
            print(f'Subsampling data with step {subsample}')
            _data_temp = evoked_f.data[:, s_min:s_max + 1:subsample]
        else:
            _data_temp = evoked_f.data[:, s_min:s_max + 1]
        temp_list = list()
        for _l in _data_temp.T:
            temp_list.append(np.vstack([np.real(_l), np.imag(_l)]).T)
        return np.hstack(temp_list), _freqs, _sfreq
    else:
        raise ValueError


def _read_fwd_ori(forward):
    fwd_ori = forward['source_ori']
    if fwd_ori == 1:
        assert (forward['sol']['data'].shape[1] == forward['source_rr'].shape[0]), \
            'Inconsistency between source space and leadfield dimensions.'
        print('Forward model with fixed source orientation.')
        return True
    elif fwd_ori == 2:
        assert (forward['sol']['data'].shape[1] == 3*forward['source_rr'].shape[0]), \
            'Inconsistency between source space and leadfield dimensions.'
        print('Forward model with free source orientation.')
        return False
    else:
        raise ValueError('Unknown source orientation in the forward model. Please check the "source_ori" '
                         'attribute of the Forward.')


def _prepare_src_lf(forward, data, noise_cov=None):
    _forward_picked, _info_picked = _select_orient_forward(forward, data.info, noise_cov)

    fixed_ori = _read_fwd_ori(_forward_picked)
    source_space = _forward_picked['source_rr']
    n_verts = source_space.shape[0]
    lead_field = _forward_picked['sol']['data']
    return _forward_picked, source_space, lead_field, n_verts, fixed_ori, _info_picked


def _compute_correlation_distance_matrix(fwd):
    print(' Computing correlation distance matrix...')
    distance_matrix = np.zeros((fwd['sol']['data'].shape[1], fwd['sol']['data'].shape[1]))
    n_ch_tot = fwd['info']['nchan']
    ch_types = set(map(lambda x: channel_type(fwd['info'], x), range(n_ch_tot)))

    for _t in ch_types:
        print('  Using {} sensors for computation...'.format(_t))
        if _t in ['mag', 'grad', 'planar1', 'planar2']:
            _fwd_t = pick_types_forward(fwd, meg=_t, ref_meg=False)
        elif _t == 'eeg':
            _fwd_t = pick_types_forward(fwd, meg=False, eeg=True, ref_meg=False)
        else:
            raise NotImplementedError

        n_ch_t = _fwd_t['info']['nchan']
        _lf_t = _fwd_t['sol']['data']
        _dm_t = ssd.cdist(_lf_t.T, _lf_t.T, metric='correlation')

        distance_matrix += (n_ch_t / n_ch_tot) * _dm_t

    print(' [done]')
    return distance_matrix


def _compute_cortical_distance_matrix(fwd):
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


def _prepare_neighbours_matrix(fwd, src, radius=None, fixed_ori=False, n_simil=1.0):
    if radius is None:
        radius = initialize_radius(src)
    else:
        radius = radius
    print('Computing neighbours matrix ', end='')
    d_matrix = ssd.cdist(src, src)
    if fixed_ori:
        if n_simil == 1:
            print('using Euclidean distance...', end='')
        elif n_simil == 0:
            print('using correlation distance...')
        else:
            print('using combined distance...')

        if n_simil == 1:
            n_matrix = compute_neighbours_matrix(src, d_matrix, radius, n_simil)
            print('[done]')
            return n_matrix, radius, None
        elif 0 <= n_simil < 1:
            correl_d_matrix = _compute_correlation_distance_matrix(fwd)
            combined_d_matrix = (1 - n_simil) * normalize(correl_d_matrix) + n_simil * normalize(d_matrix)
            n_matrix = compute_neighbours_matrix(src, combined_d_matrix, 30, n_simil)
            print('[done]')
            return n_matrix, radius, combined_d_matrix
        else:
            raise ValueError('Parameter n_simil should take values between 0 and 1. ')
    else:
        n_matrix = compute_neighbours_matrix(src, d_matrix, radius, n_simil=1)
        print('[done]')
        return n_matrix, radius, None


def _prepare_neighbours_probabilities_matrix(n_matrix, src, radius, neigh_std=None, fixed_ori=False,
                                             n_simil=1, combined_d_matrix=None):
    print('Computing neighbours probabilities...', end='')
    if neigh_std is None:
        neigh_std = radius / 2
    else:
        neigh_std = neigh_std
    d_matrix = ssd.cdist(src, src)

    if fixed_ori:
        if n_simil == 1:
            np_matrix = compute_neighbours_probability_matrix(n_matrix, src, d_matrix, neigh_std)
        elif 0 <= n_simil < 1:
            a_matrix = np.zeros(combined_d_matrix.shape, dtype=bool)
            row_list, col_list = list(), list()
            for _in, _n in enumerate(n_matrix):
                for _x in _n[_n >= 0]:
                    row_list.append(_in)
                    col_list.append(_x)
            a_matrix[(row_list, col_list)] = 1
            combined_neigh_std = np.max(combined_d_matrix[a_matrix]) / 2
            np_matrix = compute_neighbours_probability_matrix(n_matrix, src, combined_d_matrix, combined_neigh_std)
        else:
            raise ValueError

    else:
        np_matrix = compute_neighbours_probability_matrix(n_matrix, src, d_matrix, neigh_std)
    print('[done]')
    return np_matrix, neigh_std


def prepare_sesame(forward, data, n_parts=100, top_min=None, top_max=None, subsample=None,
                   noise_std=None, dip_mom_std=None,
                   prior_locs=None,
                   hyper_q=True, lam=0.25, max_n_dips=10,
                   noise_cov=None, epochs_avg=False,
                   radius=None, neigh_std=None, n_simil=0.5, fourier=False,
                   subject=None, subjects_dir=None, trans_matrix=None,
                   verbose=False):
    """ Prepare a SESAME instance for actually computing the inverse.

    Parameters
    ----------
    forward : :py:class:`~mne.Forward` object
        The forward solution.

        .. note::
            SESAME automatically detects whether the dipole orientations are free or locally normal to the
            cortical surface.
    data : instance of :py:class:`~mne.Evoked` | :py:class:`~mne.EvokedArray` | :py:class:`~mne.Epochs` | :py:class:`~mne.EpochsArray`
        The MEEG data.
    n_parts : :py:class:`~int`
        The number of particles forming the empirical pdf.
    top_min : :py:class:`~float` | None
        First topography to be included in the segment of data to be analyzed.
        It is meant to be expressed either in seconds in the time domain or
        in Hertz in the frequency domain.
        If None, it is set to the first topography of the input data.
    top_max : :py:class:`~float` | None
        Last topography to be included in the segment of data to be analyzed.
        It is meant to be expressed either in seconds in the time domain or
        in Hertz in the frequency domain.
        If None, it is set to the last topography of the input data.
    subsample : :py:class:`~int` | None
        The step used to subsample the data. If None no subsampling is
        applied.
    noise_std : :py:class:`~float` | None
        The standard deviation of the noise distribution.
        If None, it is estimated from the data.
    dip_mom_std : :py:class:`~float` | None
        The standard deviation of the prior pdf on the dipole moment.
        If None, it is estimated from the forward model and the data.
    prior_locs : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (number of points in the source space) | None
        The prior probability of source location. If None, a uniform prior probability is used.
    hyper_q : :py:class:`~bool`
        If True, a hyperprior pdf on the dipole moment std is used.
    lam : :py:class:`~float`
        The parameter of the Poisson prior pdf on the number of dipoles.
    max_n_dips : :py:class:`~int`
        The maximum allowed number of dipoles in a particle.
    noise_cov : instance of :py:class:`~mne.Covariance` | None
        The noise covariance matrix used to prewhiten the data. If None,
        no prewhitening is applied.
    epochs_avg : :py:class:`~bool`
        If True, average data epochs.
    radius : :py:class:`~float` | None
        The maximum distance in centimeters between two neighbouring vertices
        of the brain discretization. If None, it is set to 1 cm.
    neigh_std : :py:class:`~float` | None
        The standard deviation of the probability distribution of
        neighbours. If None, it is set to radius/2.
    n_simil : :py:class:`~float`
        Determines which distance is used in computing the set of neighbours of each point in the source space:

        * if ``n_simil = 0`` the correlation distance is used;

        * if ``n_simil = 1`` the Euclidean distance is used;

        * if ``0 < n_simil < 1`` a combination of the two distances is used.
          In this case, ``n_simil`` is the weight of the Euclidean distance
    fourier : :py:class:`~bool`
        If True, data are converted to the frequency domain.
    subject : :py:class:`~str` | None
        The subject name.
    subjects_dir : :py:class:`~str` | None
        If not None, this directory will be used as the subjects directory
        instead of the value set using the SUBJECTS_DIR environment variable.
    trans_matrix : instance of :py:class:`~mne.Transform` | None
        MRI<->Head coordinate transformation.
    verbose : :py:class:`~bool`
        If True, increase verbose level.


    Returns
    -------
    sesame : instance of :py:class:`~sesameeg.Sesame`
        Inverse operator
    """

    if not _is_forward(forward):
        raise ValueError('Forward must be an instance of MNE'
                         ' class Forward.')

    if not (_is_evoked(data) or _is_epochs(data)):
        raise ValueError('Data must be an instance of MNE class '
                         'Evoked, EvokedArray, Epochs or EpochsArray.')

    # src, lf
    forward, source_space, lead_field, n_verts, fixed_ori, _info = _prepare_src_lf(forward, data, noise_cov=noise_cov)

    # N, NP
    n_matrix, radius, combined_d_matrix = _prepare_neighbours_matrix(forward, source_space, radius=radius,
                                                                     fixed_ori=fixed_ori, n_simil=n_simil)
    np_matrix, neigh_std = _prepare_neighbours_probabilities_matrix(n_matrix, source_space, radius, neigh_std=neigh_std,
                                                         fixed_ori=fixed_ori, n_simil=n_simil,
                                                         combined_d_matrix=combined_d_matrix)

    # Prepare data
    if _is_evoked(data):
        _data, _times, _sfreq = _prepare_evoked_data(data, top_min, top_max, subsample=subsample, fourier=fourier)
    elif _is_epochs(data):
        _data, _times, _sfreq = _prepare_epochs_data(data, top_min, top_max, epochs_avg=epochs_avg, subsample=subsample,
                                     fourier=fourier)
    else:
        raise ValueError

    # Perform whitening if a noise covariance is provided
    if noise_cov is not None:
        if fourier:
            raise NotImplementedError('Still to implement whitening in the frequency domain')
        else:
            if _is_evoked(data):
                _sf = np.sqrt(data.nave)
            elif _is_epochs(data):
                _sf = 1.0
            else:
                raise ValueError

            whitener, _ = compute_whitener(noise_cov, info=_info, pca=True, picks=_info['ch_names'])
            _data = _sf * np.dot(whitener, _data)
            lead_field = (_sf * np.dot(whitener, lead_field))
            print('Data and leadfield have been prewhitened.')

    data_matrix = _data.real
    del _data

    # Instantiate Sesame
    if fourier is False:
        _sesame = Sesame(source_space, lead_field, data_matrix, n_parts=n_parts, n_matrix=n_matrix,
                         np_matrix=np_matrix, noise_std=noise_std, dip_mom_std=dip_mom_std,
                         fixed_ori=fixed_ori, radius=radius, neigh_std=neigh_std, prior_locs=prior_locs,
                         subsample=subsample, hyper_q=hyper_q, lam=lam, max_n_dips=max_n_dips,
                         fourier=fourier, verbose=verbose, forward=forward, subject=subject,
                         subjects_dir=subjects_dir, data_times=_times, data_sfreq=_sfreq, trans_matrix=trans_matrix)
    else:
        _sesame = Sesame(source_space, lead_field, data_matrix, n_parts=n_parts, n_matrix=n_matrix,
                         np_matrix=np_matrix, noise_std=noise_std, dip_mom_std=dip_mom_std,
                         fixed_ori=fixed_ori, radius=radius, neigh_std=neigh_std, prior_locs=prior_locs,
                         subsample=subsample, hyper_q=hyper_q, lam=lam, max_n_dips=max_n_dips,
                         fourier=fourier, verbose=verbose, forward=forward, subject=subject,
                         subjects_dir=subjects_dir, data_freqs=_times, data_sfreq=_sfreq, trans_matrix=trans_matrix)

    return _sesame
