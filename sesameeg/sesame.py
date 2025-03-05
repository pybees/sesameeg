# -*- coding: utf-8 -*-

# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#          Alberto Sorrentino <sorrentino@dima.unige.it>
#
# License: BSD (3-clause)

import os
os.environ['SCIPY_ARRAY_API'] = '1'
import itertools
import time
import copy
import numpy as np
import scipy.spatial.distance as ssd
from mne.evoked import EvokedArray
from .emp_pdf import EmpPdf
from .particles import Particle
from .utils import compute_neighbours_matrix, initialize_radius, \
    compute_neighbours_probability_matrix, estimate_noise_std, estimate_dip_mom_std
from .io import _export_to_stc, _export_to_vol_stc, write_h5, write_pkl
from .viz import plot_n_sources, plot_amplitudes, plot_stc, plot_vol_stc, plot_cloud_sources
from .metrics import compute_goodness_of_fit, compute_sd


class Sesame(object):
    """Sequential Semi-Analytic Monte-Carlo Estimator (SESAME).

    Parameters
    ----------
    source_space : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts, 3)
        The coordinates of the points in the brain discretization.
    lead_field : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_sens, n_comp*n_verts)
        The lead field matrix. (if ``fixed_ori=True`` it must be ``n_comp = 1``;  if ``fixed_ori=False``
        it must be ``n_comp = 3``)
    data : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_sens, n_ist)
        The MEEG data; ``n_sens`` is the number of sensors and
        ``n_ist`` is the number of time-points or of frequencies.
    n_parts : :py:class:`~int`
        The number of particles forming the empirical pdf.
    s_min : :py:class:`~int`
        The first analyzed sample in the data array.
    s_max : :py:class:`~int`
        The last analyzed sample in the data array.
    n_matrix : :py:class:`~numpy.ndarray` of :py:class:`~int`, shape (n_verts, n_max_neigh) | None
        The set of neighbours of each point in the brain discretization. If None, it is automatically computed.
        ``n_max_neigh`` is the cardinality of the biggest set.
    np_matrix : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts, n_max_neigh) | None
        The neighbours' probabilities. If None, it is automatically computed.
    noise_std : :py:class:`~float` | None
        The standard deviation of the noise distribution.
        If None, it is estimated from the data.
    dip_mom_std : :py:class:`~float` | None
        The standard deviation of the prior pdf on the dipole moment.
        If None, it is estimated from the forward model and the data.
    fixed_ori : :py:class:`~bool`
        If True, the forward solution is assumed to be computed in the surface-based source coordinate system.
    radius : :py:class:`~float` | None
        The maximum distance in cm between two neighbouring vertices
        of the brain discretization. If None, ``radius`` is set to 1 cm.
    neigh_std : :py:class:`~float` | None
        The standard deviation of the probability distribution of
        neighbours. If None, ``neighb_std`` is set to radius/2.
    prior_locs : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_vert) | None
        The prior probability of source location. If None, a uniform prior probability is used.
        Default is None.
    subsample : :py:class:`~int` | None
        The step used to subsample the data. If None no subsampling is
        applied.
    hyper_q : :py:class:`~bool`
        If True, a hyperprior pdf on the dipole moment std will be used.
    lam : :py:class:`~float`
        The parameter of the Poisson prior pdf on the number of dipoles.
    max_n_dips : :py:class:`~int`
        The maximum number of dipoles allowed in a particle.
    fourier : :py:class:`~bool`
        If True, data are converted to the frequency domain.
    verbose : :py:class:`~bool`
        If True, increase verbose level.
    **kwargs :
        Additional keyword arguments are passed to the Sesame constructor.

    Attributes
    ----------
    dip_mom_std : :py:class:`~float`
        The standard deviation of the prior on the dipole moment.
    distance_matrix : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts x n_verts)
        The Euclidean distance between the points in the
        brain discretization.
    est_dip_moms : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_ist x (n_comp*est_n_dips[-1])) | None
        The sources' moments estimated at the last iteration. (n_comp = 1, if fixed orientation, 3, if free orientation)
        If None, moments can be estimated by calling :py:meth:`~Sesame.compute_dip_mom`
    est_locs : :py:class:`~list` of :py:class:`~numpy.ndarray` of :py:class:`~int`
        The source space grid points indices in which a source is estimated.
    est_n_dips : :py:class:`~list` of :py:class:`~int`
        The estimated number of dipoles.
    fourier : :py:class:`~bool`
        If True, data are in the frequency domain.
    hyper_q : :py:class:`~bool`
        If True use hyperprior on dipole moment std.
    lam : :py:class:`~float`
        The parameter of the Poisson prior pdf on the number of dipoles.
    lead_field : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_sens x n_comp*n_verts)
        The leadfield matrix. (``n_comp = 1`` if ``fixed_ori=True``; ``n_comp = 3`` if ``fixed_ori=False``)
    max_n_dips : :py:class:`~int`
        The maximum number of dipoles allowed in a particle.
    model_sel : :py:class:`~list` of :py:class:`~numpy.ndarray` of :py:class:`~float`
        The model selection, i.e. the posterior distribution on the number
        of dipoles.
    n_verts : :py:class:`~int`
        The number of points forming the brain discretization.
    n_parts : :py:class:`~int`
        The number of particles forming the empirical pdf.
    neigh : :py:class:`~numpy.ndarray` of :py:class:`~int`, shape (n_vert, n_max_neigh)
        The set of neighbours of each point in the brain discretization.
        n_max_neigh is the cardinality of the biggest set.
    neigh_p : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_vert, n_max_neigh)
        The neighbours' probabilities.
    neigh_std : :py:class:`~float`
        The standard deviation used to compute the neigh_p matrix.
    noise_std : :py:class:`~float`
        The standard deviation of the noise distribution.
    pmap : :py:class:`~list` of :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (est_n_dips, n_verts)
        Posterior probability map.
    posterior : instance of :class:`~sesameeg.emppdf.EmpPdf`
        The empirical pdf approximated by the particles at each iteration.
    prior_locs : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts, ) | None
        The prior probability of active source locations. If None, each source space grid point is assigned a
        uniform prior probability.
    r_data : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_sens, n_ist)
        The real part of the data; n_sens is the number of sensors and
        n_ist is the number of time-points or of frequencies.
    radius : :py:class:`~float`
        The radius used to compute the neighbours.
    _resample_it : :py:class:`~list` of :py:class:`~int`
        The iterations during which a resampling step has been performed
    s_min : :py:class:`~int`
        The first sample of the segment of data that are analyzed.
    s_max : :py:class:`~int`
        The last sample of the segment of data that are analyzed.
    source_space : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts, 3)
        The coordinates of the points in the brain discretization.
    subsample : :py:class:`~int` | None
        The step used to subsample the data.

    data_times : :py:class:`~numpy.ndarray`
        Time vector in seconds. Only when instantiated by means of :func:`~sesameeg.mne.prepare_sesame`
        and when ``fourier=False``
    data_freqs : :py:class:`~numpy.ndarray`
        Frequency vector in Hertz. Only when instantiated by means of :func:`~sesameeg.mne.prepare_sesame`
        and when ``fourier=True``
    forward : instance of :py:class:`~mne.Forward`
        The forward solution. Only when instantiated by means of :func:`~sesameeg.mne.prepare_sesame`
    subject : :py:class:`~str`
        Subject name in Freesurfer subjects dir. Only when instantiated by means
        of :func:`~sesameeg.mne.prepare_sesame`.
    subjects_dir : :py:class:`~str` | None
        If not None, this directory will be used as the subjects directory instead of the value set using
        the SUBJECTS_DIR environment variable. Only when instantiated by means
        of :func:`~sesameeg.mne.prepare_sesame`
    trans_matrix : instance of :py:class:`~mne.Transform`
        MRI<->Head coordinate transformation. Only when instantiated by
        means of :func:`~sesameeg.mne.prepare_sesame`
    """

    def __init__(self, source_space, lead_field, data,
                 n_parts=100,
                 s_min=None,
                 s_max=None,
                 n_matrix=None,
                 np_matrix=None,
                 noise_std=None,
                 dip_mom_std=None,
                 fixed_ori=False,
                 radius=None,
                 neigh_std=None,
                 prior_locs=None,
                 subsample=None,
                 hyper_q=True,
                 lam=0.25,
                 max_n_dips=10,
                 fourier=False,
                 verbose=False,
                 **kwargs):

        # 1) Chosen by the user
        self.fourier = fourier
        self.n_parts = n_parts
        self.lam = lam
        self.max_n_dips = max_n_dips
        self.s_min = 0 if s_min is None else s_min
        self.s_max = (data.shape[1] - 1) if s_max is None else s_max
        self.subsample = subsample
        self.verbose = verbose
        self.hyper_q = hyper_q
        self.fixed_ori = fixed_ori
        self.source_space = source_space
        self.stc = None
        self.n_verts = self.source_space.shape[0]
        self.lead_field = lead_field
        self.distance_matrix = ssd.cdist(self.source_space, self.source_space)
        self.prior_locs = self._get_prior_locs(prior_locs, self.n_verts)
        # assert(np.sum(self.prior_locs == 1))

        # Kwargs
        self.forward = None
        self.trans_matrix = None
        self.subject = None
        self.subjects_dir = None
        self.data_times = None
        self.data_freqs = None
        self.data_sfreq = None
        for k, v in kwargs.items():
            if k in self.__dict__:
                setattr(self, k, v)
            else:
                raise KeyError(k)

        if n_matrix is None:
            print('Computing neighbours matrix ', end='')
            if radius is None:
                self.radius = initialize_radius(self.source_space)
            else:
                self.radius = radius
            self.neigh = compute_neighbours_matrix(self.source_space,
                                                   self.distance_matrix,
                                                   self.radius,
                                                   n_simil=1)
            print('[done]')
        else:
            self.neigh = n_matrix

        if np_matrix is None:
            print('Computing neighbours probabilities...', end='')
            if neigh_std is None:
                self.neigh_std = self.radius / 2
            else:
                self.neigh_std = neigh_std
            self.neigh_p = compute_neighbours_probability_matrix(self.neigh,
                                                                 self.source_space,
                                                                 self.distance_matrix,
                                                                 self.neigh_std)
            print('[done]')
        else:
            self.neigh_p = np_matrix

        #  Data
        if subsample is not None:
            print(f'Subsampling data with step {subsample}')
            self.r_data = data[:, self.s_min:self.s_max + 1:self.subsample]
            if self.data_times is not None:
                self.data_times = self.data_times[self.s_min:self.s_max + 1:self.subsample]
            elif self.data_freqs is not None:
                self.data_freqs = self.data_freqs[self.s_min:self.s_max + 1:self.subsample]
        else:
            self.r_data = data[:, self.s_min:self.s_max + 1]
            if self.data_times is not None:
                self.data_times = self.data_times[self.s_min:self.s_max + 1]
            elif self.data_freqs is not None:
                self.data_freqs = self.data_freqs[self.s_min:self.s_max + 1]

        # Dipole moment std
        if dip_mom_std is None:
            print('Estimating dipole moment std...', end='')
            self.dip_mom_std = estimate_dip_mom_std(self.r_data, self.lead_field)
            print('[done]')
            print(' Estimated dipole moment std: {:.4e}'
                  .format(self.dip_mom_std))
        elif isinstance(dip_mom_std, float):
            self.dip_mom_std = dip_mom_std
            print('User defined dipole moment std: {:.4e}'
                  .format(self.dip_mom_std))
        else:
            raise ValueError('Dipole moment std must be either None or a float.')

        if self.hyper_q:
            print('Sampling hyperprior for dipole moment std.')

        # Noise std
        if noise_std is None:
            print('Estimating noise std...', end='')
            self.noise_std = estimate_noise_std(self.r_data)
            print('[done]')
            print(' Estimated noise std: {:.4e}'.format(self.noise_std))
        else:
            self.noise_std = noise_std
            print('User defined noise std: {:.4e}'
                  .format(self.noise_std))

        self._resample_it = list()
        self.model_sel = list()
        self.pmap = list()
        self.est_n_dips = list()
        self.est_locs = list()
        self.est_dip_moms = None
        self.est_dip_mom_std = list()

        self.posterior = EmpPdf(self.n_parts, self.n_verts, self.lam, dip_mom_std=self.dip_mom_std,
                                prior_locs=self.prior_locs, fixed_ori=self.fixed_ori,
                                hyper_q=self.hyper_q, verbose=self.verbose)

        for _part in self.posterior.particles:
            if self.hyper_q:
                _aux = 0
                _pos_def = _part._check_sigma(self.r_data, self.lead_field,
                                              self.noise_std)
                while _pos_def is False:
                    _part.dip_mom_std = 10 ** (3 * np.random.rand()) * (self.dip_mom_std / 35)
                    _pos_def = _part._check_sigma(self.r_data, self.lead_field,
                                                  self.noise_std)
                    _aux += 1
                    if _aux == 100:
                        raise ValueError

            _part.compute_loglikelihood_unit(self.r_data, self.lead_field,
                                             noise_std=self.noise_std)

    @staticmethod
    def _get_prior_locs(p_locs, n_verts):
        _error = 'Prior source location probabilities should be given as a one dimensional ' \
                 'array of the same shape as the source space.'
        if p_locs is None:
            pl_array = np.ones(n_verts)
            return pl_array / np.sum(pl_array)
        else:
            if isinstance(p_locs, np.ndarray):
                print('Sampling user defined prior probability distribution for dipole locations.')
                if (p_locs.ndim != 1) or (p_locs.shape[0] != n_verts):
                    raise ValueError(_error)
                return p_locs / np.sum(p_locs)
            else:
                raise ValueError(_error)

    def _get_topographies(self, evoked, top_min, top_max, freqs=None):
        if self.fourier is False:
            if top_min is None:
                self.s_min = 0
                self.top_min = evoked.times[0]
            else:
                if isinstance(top_min, (float, np.float)):
                    self.s_min = evoked.time_as_index(top_min, use_rounding=True)[0]
                    self.top_min = evoked.times[self.s_min]
                else:
                    raise ValueError('top_min value should be a float')

            if top_max is None:
                self.s_max = evoked.data.shape[1] - 1
                self.top_max = evoked.times[-1]
            else:
                if isinstance(top_max, (float, np.float)):
                    self.s_max = evoked.time_as_index(top_max, use_rounding=True)[0]
                    self.top_max = evoked.times[self.s_max]
                else:
                    raise ValueError('top_max value should be a float')
            print('Analyzing data from {0} s to {1} s'.format(round(self.top_min, 4), round(self.top_max, 4)))
        elif self.fourier is True:
            if top_min is None:
                self.s_min = 0
                self.top_min = freqs[0]
            else:
                if isinstance(top_min, (float, np.float)):
                    self.s_min = np.where((freqs >= top_min))[0][0]
                    self.top_min = freqs[self.s_min]
                else:
                    raise ValueError('top_min value should be a float')

            if top_max is None:
                self.s_max = evoked.data.shape[1] - 1
                self.top_max = freqs[-1]
            else:
                if isinstance(top_max, (float, np.float)):
                    self.s_max = np.where((freqs <= top_max))[0][-1]
                    self.top_max = freqs[self.s_max]
                else:
                    raise ValueError('top_max value should be a float')
            print('Analyzing data from {0} Hz to {1} Hz'.format(round(self.top_min, 4), round(self.top_max, 4)))

    def _prepare_epochs_data(self, epochs, top_min, top_max, epochs_avg):
        ep_data = epochs.get_data()
        evoked = EvokedArray(ep_data[0], epochs.info, tmin=epochs.times[0])
        if self.fourier is False:
            self._get_topographies(evoked, top_min, top_max)
            temp_list = list()
            for ie, _e in enumerate(ep_data):
                if self.subsample is not None:
                    if ie == 0:
                        print('Subsampling data with step {0}'.format(self.subsample))
                    temp_list.append(_e[:, self.s_min:self.s_max + 1:self.subsample])
                else:
                    temp_list.append(_e[:, self.s_min:self.s_max + 1])
            if epochs_avg is True:
                print(f'Averaging {len(temp_list)} epochs.')
                return np.mean(np.array(temp_list), axis=0)
            else:
                return np.hstack(temp_list)
        elif self.fourier is True:
            tstep = 1 / evoked.info['sfreq']
            evoked_f = evoked.copy()
            evoked_f.data *= np.hamming(evoked.data.shape[1])
            evoked_f.data = (np.fft.rfft(evoked_f.data))
            freqs = np.fft.rfftfreq(evoked.data.shape[1], tstep)
            print('Data have been converted to the frequency domain.')

            self._get_topographies(evoked_f, top_min, top_max, freqs=freqs)

            temp_list = list()
            temp_list2 = list()
            temp_list2_r = list()
            temp_list2_i = list()
            for ie, _e in enumerate(ep_data):
                _e *= np.hamming(_e.shape[1])
                _e_f = np.fft.rfft(_e)
                if self.subsample is not None:
                    if ie == 0:
                        print('Subsampling data with step {0}'.format(self.subsample))
                    temp_list.append(_e_f[:, self.s_min:self.s_max + 1:self.subsample])
                else:
                    temp_list.append(_e_f[:, self.s_min:self.s_max + 1])
                self._tl = temp_list

            for _data_temp in temp_list:
                temp_list2.append(np.real(_data_temp))
                temp_list2.append(np.imag(_data_temp))

                temp_list2_r.append(np.real(_data_temp))
                temp_list2_i.append(np.imag(_data_temp))

                #for _l in _data_temp.T:
                #    temp_list2.append(np.vstack([np.real(_l), np.imag(_l)]).T)
            if epochs_avg is True:
                print(f'Averaging {len(temp_list)} epochs.')
                _r_mean = np.mean(np.array(temp_list2_r), axis=0)
                _i_mean = np.mean(np.array(temp_list2_i), axis=0)

                return np.hstack([_r_mean, _i_mean])
            else:
                return np.hstack(temp_list2)
        else:
            raise ValueError

    def _prepare_evoked_data(self, evoked, top_min, top_max):
        if self.fourier is False:
            self._get_topographies(evoked, top_min, top_max)

            if self.subsample is not None:
                print('Subsampling data with step {0}'.format(self.subsample))
                _data = evoked.data[:, self.s_min:self.s_max + 1:self.subsample]
            else:
                _data = evoked.data[:, self.s_min:self.s_max + 1]
            return _data
        elif self.fourier is True:
            tstep = 1 / evoked.info['sfreq']
            evoked_f = evoked.copy()
            evoked_f.data *= np.hamming(evoked.data.shape[1])
            evoked_f.data = (np.fft.rfft(evoked_f.data))
            freqs = np.fft.rfftfreq(evoked.data.shape[1], tstep)
            print('Data have been converted to the frequency domain.')

            self._get_topographies(evoked_f, top_min, top_max, freqs=freqs)

            if self.subsample is not None:
                print('Subsampling data with step {0}'.format(self.subsample))
                _data_temp = evoked_f.data[:, self.s_min:self.s_max + 1:self.subsample]
            else:
                _data_temp = evoked_f.data[:, self.s_min:self.s_max + 1]
            temp_list = list()
            for _l in _data_temp.T:
                temp_list.append(np.vstack([np.real(_l), np.imag(_l)]).T)
            return np.hstack(temp_list)
        else:
            raise ValueError

    def _read_fwd_ori(self):
        fwd_ori = self.forward['source_ori']
        if fwd_ori == 1:
            assert (self.forward['sol']['data'].shape[1] == self.forward['source_rr'].shape[0]), \
                'Inconsistency between source space and leadfield dimensions.'
            print('Forward model with fixed source orientation.')
            return True
        elif fwd_ori == 2:
            assert (self.forward['sol']['data'].shape[1] == 3*self.forward['source_rr'].shape[0]), \
                'Inconsistency between source space and leadfield dimensions.'
            print('Forward model with free source orientation.')
            return False
        else:
            raise ValueError('Unknown source orientation in the forward model. Please check the "source_ori" '
                             'attribute of the Forward.')

    def _reset_attributes(self):
        self._resample_it = list()
        self.est_n_dips = list()
        self.est_locs = list()
        self.est_dip_moms = None
        self.est_dip_mom_std = None
        self.final_dip_mom_std = None
        self.model_sel = list()
        self.pmap = list()

        if self.hyper_q:
            self.est_dip_mom_std = list(np.array([]) for _ in range(self.n_parts))

        self.posterior = EmpPdf(self.n_parts, self.n_verts, self.lam, dip_mom_std=self.dip_mom_std,
                                prior_locs=self.prior_locs, fixed_ori=self.fixed_ori,
                                hyper_q=self.hyper_q, verbose=self.verbose)

        for _part in self.posterior.particles:
            if self.hyper_q:
                _aux = 0
                _pos_def = _part._check_sigma(self.r_data, self.lead_field,
                                              self.noise_std)
                while _pos_def is False:
                    _part.dip_mom_std = 10 ** (3 * np.random.rand()) * (self.dip_mom_std / 35)
                    _pos_def = _part._check_sigma(self.r_data, self.lead_field,
                                                  self.noise_std)
                    _aux += 1
                    if _aux == 100:
                        raise ValueError

            _part.compute_loglikelihood_unit(self.r_data, self.lead_field,
                                             noise_std=self.noise_std)

    def apply_sesame(self, estimate_all=False, estimate_dip_mom=True):
        """Apply SESAME on MEEG data and compute point estimates.

        Parameters
        ----------
        estimate_all : :py:class:`~bool`
            If True compute the posterior probability map and estimate the
            number of dipoles and their locations at each iteration.
            If False compute the above quantities only at the last
            iteration.
        estimate_dip_mom : :py:class:`~bool`
            If True compute a point-estimate of the dipole moment at the
            last iteration.
        """

        if self.posterior.exponents[-1] > 0:
            print('Resetting SESAME...', end='')
            self._reset_attributes()
            print('[done]')

        print('Computing inverse solution. This will take a while...')
        # --------------- INITIALIZATION ---------------
        # Samples are drawn from the prior distribution and weigths are set as
        # uniform.
        nd = np.array([_part.n_dips for _part in self.posterior.particles])

        while not np.all(nd <= self.max_n_dips):
            nd_wrong = np.where(nd > self.max_n_dips)[0]
            self.posterior.particles[nd_wrong] =\
                np.array([Particle(self.n_verts, self.lam, dip_mom_std=self.dip_mom_std)
                         for _ in itertools.repeat(None, nd_wrong.shape[0])])
            nd = np.array([_part.n_dips for _part in self.posterior.particles])

        # Point estimation for the first iteration
        if estimate_all:
            self.posterior.point_estimate(self.distance_matrix, self.max_n_dips)

            self.est_n_dips.append(self.posterior.est_n_dips)
            self.model_sel.append(self.posterior.model_sel)
            self.est_locs.append(self.posterior.est_locs)
            self.pmap.append(self.posterior.pmap)
            if self.hyper_q:
                self.est_dip_mom_std.append(self.posterior.est_dip_mom_std)

        # --------------- MAIN CYCLE ---------------

        while np.all(self.posterior.exponents <= 1):
            time_start = time.time()
            if self.verbose:
                print('iteration = {0}'.format(self.posterior.exponents.shape[0]))
                print('exponent = {0}'.format(self.posterior.exponents[-1]))
                print('ESS = {:.2%}'.format(self.posterior.ESS/self.n_parts))

            # STEP 1: (possible) resampling
            if self.posterior.ESS < self.n_parts/2:
                self._resample_it.append(int(self.posterior.exponents.shape[0]))
                self.posterior.resample()
                if self.verbose:
                    print('----- RESAMPLING -----')
                    print('ESS = {:.2%}'.format(self.posterior.ESS/self.n_parts))

            # STEP 2: Sampling.
            self.posterior.sample(self.n_verts, self.r_data, self.lead_field,
                                  self.neigh, self.neigh_p, self.noise_std,
                                  self.lam, self.max_n_dips)

            # STEP 3: Point Estimation
            if estimate_all:
                self.posterior.point_estimate(self.distance_matrix, self.max_n_dips)

                self.est_n_dips.append(self.posterior.est_n_dips)
                self.model_sel.append(self.posterior.model_sel)
                self.est_locs.append(self.posterior.est_locs)
                self.pmap.append(self.posterior.pmap)
                if self.hyper_q:
                    self.est_dip_mom_std.append(self.posterior.est_dip_mom_std)

            # STEP 4: compute new exponent and new weights
            self.posterior.compute_exponent(self.noise_std)

            time.sleep(0.01)
            time_elapsed = (time.time() - time_start)
            if self.verbose:
                print('Computation time: {:.2f} seconds'.format(time_elapsed))
                print('-------------------------------')

        # Point estimation
        self.posterior.point_estimate(self.distance_matrix, self.max_n_dips)

        self.est_n_dips.append(self.posterior.est_n_dips)
        self.model_sel.append(self.posterior.model_sel)
        self.est_locs.append(self.posterior.est_locs)
        self.pmap.append(self.posterior.pmap)
        if self.hyper_q:
            self.est_dip_mom_std.append(self.posterior.est_dip_mom_std)
            print('Estimated dipole strength variance: {}'.format(self.est_dip_mom_std[-1]))

        if estimate_dip_mom:
            if self.est_n_dips[-1] == 0:
                self.est_dip_moms = np.array([])
            else:
                self.compute_dip_mom(self.est_locs[-1])

        # Print results
        print(f'    Estimated number of sources: {self.est_n_dips[-1]}')
        if self.est_n_dips[-1] > 0:
            print('    Estimated source locations:')
            for _iloc, _loc in enumerate(self.est_locs[-1]):
                print(f'        * source {_iloc + 1}: {self.source_space[_loc]}')
        print(f'[done in {self.posterior.exponents.shape[0]} iterations]')

    def compute_dip_mom(self, est_locs):
        """Compute a point estimate for the dipole moments.

        Parameters
        ----------
        est_locs : :py:class:`~list` of :py:class:`~numpy.ndarray` of :py:class:`~int`
            The estimated source locations.
        """
        est_num = est_locs.shape[0]
        [n_sens, n_time] = np.shape(self.r_data)

        if self.hyper_q:
            _dip_mom_std = self.est_dip_mom_std[-1]
        else:
            _dip_mom_std = self.dip_mom_std

        if self.fixed_ori:
            ind = np.ravel([est_locs[idip] for idip in range(est_num)])
        else:
            ind = np.ravel([[3 * est_locs[idip], 3 * est_locs[idip] + 1,
                             3 * est_locs[idip] + 2] for idip in range(est_num)])

        Gc = self.lead_field[:, ind]
        sigma = (_dip_mom_std / self.noise_std)**2 * np.dot(Gc, np.transpose(Gc)) +\
            np.eye(n_sens)
        kal_mat = (_dip_mom_std / self.noise_std)**2 * np.dot(np.transpose(Gc),
                                                        np.linalg.inv(sigma))
        self.est_dip_moms = np.array([np.dot(kal_mat, self.r_data[:, t]) for t in range(n_time)])

    def goodness_of_fit(self):
        """Evaluate the estimated configuration of dipoles. The goodness
        of fit (GOF) with the recorded data is defined as

        .. math:: GOF = \\frac{\\| \\mathbf{y} - \\hat{\\mathbf{y}} \\|}
                         { \\|\\mathbf{y}\\|}

        where :math:`\\mathbf{y}` is the recorded data,
        :math:`\\hat{\\mathbf{y}}` is the field generated by the
        estimated configuration of dipoles, and :math:`\\| \\cdot \\|`
        is the Frobenius norm.

        Returns
        -------
        gof : :py:class:`~float`
            The goodness of fit with the recorded data.
        """

        if self.est_n_dips[-1] == 0:
            raise AttributeError("No dipole has been estimated."
                                 "Run apply_sesame first and set noise_std properly.")
        if self.est_dip_moms is None:
            raise AttributeError("No dipoles' moment found."
                                 " Run compute_q first.")

        gof = compute_goodness_of_fit(self.r_data, self.est_n_dips[-1],
                                      self.est_locs[-1], self.est_dip_moms,
                                      self.lead_field)
        return gof

    def source_dispersion(self):
        """Compute the Source Dispersion measure to quantify the spatial dispersion
        of the posterior probability map. It is defined as

         .. math:: SD = \\sqrt{\\frac{\\sum_{j=1}^{N_v} \\big(d_j |S_j|\\big)^2}{\\sum_{j=1}^{N_v}|S_j|^2}}

        where :math:`N_v` is the number of voxels, :math:`d_j` is the distance between the :math:`j`-th voxel
        and the nearest estimated dipole location and :math:`S_j` is the value of the cortical map at
        the :math:`j`-th voxel.

        Returns
        -------
        sd : :py:class:`~float`
            The Source Dispersion of SESAME result
        """
        if self.est_n_dips[-1] == 0:
            raise AttributeError("No dipole has been estimated."
                                 "Run apply_sesame first and set noise_std properly.")

        pmap_tot = np.sum(self.pmap[-1], axis=0)
        est_pos = self.source_space[self.est_locs[-1]]
        sd = compute_sd(self.source_space, pmap_tot, est_pos)
        return sd

    def compute_stc(self):
        """Compute and export in an .stc file the posterior pdf
            :math:`p(r|\\mathbf{y}, \\hat{n}_D)`, being :math:`\\hat{n}_D`
            the estimated number of sources.
            For each point :math:`r` in the brain discretization,
            :math:`p(r|\\mathbf{y}, \\hat{n}_D)` is the probability of a
            source being located in :math:`r`.

            Parameters
            ----------
            subject : :py:class:`~str` | None
                The subject name.

            Returns
            -------
            stc : :py:class:`~mne.SourceEstimate` | :py:class:`~mne.VolSourceEstimate`
                The source estimate object containing the posterior map of the
                dipoles' location.
            """
        if self.forward is not None:
            if self.forward['src'].kind == 'surface':
                print('Surface stc computed.')
                return _export_to_stc(self, subject=self.subject)
            elif self.forward['src'].kind == 'volume':
                print('Volume stc computed  ')
                return _export_to_vol_stc(self, subject=self.subject)
            else:
                raise ValueError('src can be either surface or volume')
        else:
            raise AttributeError('This method works only within MNE-Python environment. '
                                 'Use sesameeg.mne.prepare_sesame to instantiate the inverse operator.')

    def plot_source_number(self, kind='bar'):
        """
        Plot the probability of number of sources.

        Parameters
        ----------
        kind : :py:class:`~str`
            The kind of plot to draw. Options are: “bar”, “pie”.
        """
        plot_n_sources(self, kind=kind, title=self.subject)

    def plot_source_amplitudes(self, n_sources=None):
        """
        Plot the  amplitude of the estimated sources as function of time.

        Parameters
        ----------
        n_sources: :py:class:`~int` | None
            Set the number of sources of the alternative configuration to plot.
            If None, plot the configuration with the estimated number of sources.
        """
        if n_sources is None:
            inv_op = self
        else:
            inv_op = copy.deepcopy(self)
            if inv_op.model_sel[-1][n_sources] > 0:
                inv_op.posterior.point_estimate(inv_op.distance_matrix, inv_op.max_n_dips,
                                                n_sources=n_sources)
                inv_op.est_n_dips = [n_sources]
                inv_op.est_locs = [inv_op.posterior.est_locs]
                inv_op.pmap = [inv_op.posterior.pmap]
                inv_op.compute_dip_mom(inv_op.est_locs[-1])
            else:
                print('The selected configuration has zero probability! Nothing to plot.')
                return

        plot_amplitudes(inv_op, title=inv_op.subject)

    def plot_sources(self, n_sources=None, plot_kwargs=None, savepath=None, save_kwargs=None,
                     true_sources=None, force_open=False):
        """
        Plot the estimated sources. The default behaviour of the method is the following:

        * if ``Sesame`` has been instantiated through :py:meth:`~mne.prepare_sesame`, it visualizes the posterior map of the dipoles’ location and the estimated sources

          * on the inflated brain, if ``Sesame.forward`` is of kind ``surface``;

          * on the MRI, if ``Sesame.forward`` is of kind ``volume``.

        * if ``Sesame`` has been instantiated through :py:class:`~Sesame`, it visualizes the posterior map of the dipoles’ location and the estimated sources as a :py:class:`~pyvista.PolyData` object.

        Parameters
        ----------
        n_sources: :py:class:`~int` | None
            Set the number of sources of the alternative configuration to plot.
            If None, plot the configuration with the estimated number of sources.
        plot_kwargs : :py:class:`~dict` | None
            Additional arguments to :py:func:`~mne.viz.plot_source_estimates` or
            :py:func:`~nilearn.plotting.plot_stat_map` (e.g., dict(surface='white')).
        savepath : :py:class:`~str` | None
            File path to write image to. If None, no image is written.
        save_kwargs : :py:class:`~dict` | None
            Additional arguments to :py:meth:`~pyvista.Plotter.screenshot`
            or :py:func:`~matplotlib.pyplot.savefig`.
        true_sources : :py:class:`~numpy.ndarray` | None
            In simulation settings, indexes of source space points in which true sources are located.
        force_open : :py:class:`~bool`
            If True, force the image to stay open.
        """
        if n_sources is None:
            inv_op = self
        else:
            inv_op = copy.deepcopy(self)
            if inv_op.model_sel[-1][n_sources] > 0:

                inv_op.posterior.point_estimate(inv_op.distance_matrix, inv_op.max_n_dips,
                                                n_sources=n_sources)
                # Normalize pmap
                _sum_aux = inv_op.posterior.pmap.sum() / n_sources
                inv_op.posterior.pmap /= _sum_aux
                # Set estimate values
                inv_op.est_n_dips = [n_sources]
                inv_op.est_locs = [inv_op.posterior.est_locs]
                inv_op.pmap = [inv_op.posterior.pmap]

                # Print results
                print(f'    Selected number of sources: {inv_op.est_n_dips[-1]}')
                print('    Estimated source locations:')
                for _iloc, _loc in enumerate(inv_op.est_locs[-1]):
                    print(f'        * source {_iloc + 1}: {inv_op.source_space[_loc]}')
            else:
                print('The selected configuration has zero probability! Nothing to plot.')
                return

        if inv_op.forward is not None:
            inv_op.stc = inv_op.compute_stc()
            stc_kind = type(inv_op.stc).__name__
            if stc_kind == 'SourceEstimate':
                plot_stc(inv_op, true_idxs=true_sources, savepath=savepath,
                         plot_kwargs=plot_kwargs, save_kwargs=save_kwargs, force_open=force_open)
            elif stc_kind == 'VolSourceEstimate':
                plot_vol_stc(inv_op, savepath=savepath, plot_kwargs=plot_kwargs,
                             save_kwargs=save_kwargs)
        else:
            plot_cloud_sources(inv_op, true_idxs=true_sources, savepath=savepath)

        del inv_op

    def save_h5(self, fpath, sbj=None, sbj_viz=None, data_path=None,
                fwd_path=None, cov_path=None, src_path=None, lf_path=None):
        """Save SESAME result to an HDF5 file.

        Parameters
        ----------
        fpath : :py:class:`~str`
            The path to the save file.
        sbj : :py:class:`~str` | None
            The subject name.
        sbj_viz : :py:class:`~str` | None
            The name of the subject's FreeSurfer folder.
        data_path : :py:class:`~str` | None
            The path to the data file.
        fwd_path : :py:class:`~str` | None
            The path to the forward solution file.
        cov_path : :py:class:`~str` | None
            The path to the noise covariance file.
        src_path : :py:class:`~str` | None
            The path to the source space file.
        lf_path : :py:class:`~str` | None
            The path to the leadfield matrix file.
        """

        h5_kwargs = dict(smin=self.s_min, smax=self.s_max, subsample=self.subsample,
                         sbj=sbj, sbj_viz=sbj_viz, data_path=data_path, fwd_path=fwd_path,
                         src_path=src_path, lf_path=lf_path)

        if self.data_times is not None:
            h5_kwargs.update(data_times=self.data_times)

        if self.data_freqs is not None:
            h5_kwargs.update(data_freqs=self.data_freqs)

        write_h5(fpath, self, **h5_kwargs)


        # if self.fourier is False:
        #     write_h5(fpath, self, tmin=self.top_min, tmax=self.top_max,
        #              subsample=self.subsample, sbj=sbj, sbj_viz=sbj_viz,
        #              data_path=data_path, fwd_path=fwd_path, src_path=src_path,
        #              lf_path=lf_path)
        # else:
        #     write_h5(fpath, self, fmin=self.top_min, fmax=self.top_max,
        #              subsample=self.subsample, sbj=sbj, sbj_viz=sbj_viz,
        #              data_path=data_path, fwd_path=fwd_path, src_path=src_path,
        #              lf_path=lf_path)

    def save_pkl(self, fpath, sbj=None, sbj_viz=None, data_path=None,
                 fwd_path=None, cov_path=None, src_path=None,
                 lf_path=None, save_all=False):
        """Save SESAME result to a Python pickle file.

        Parameters
        ----------
        fpath : :py:class:`~str`
            The path to the save file.
        sbj : :py:class:`~str` | None
            The subject name.
        sbj_viz : :py:class:`~str` | None
            The name of the subject's FreeSurfer folder.
        data_path : :py:class:`~str` | None
            The path to the data file.
        fwd_path : :py:class:`~str` | None
            The path to the forward solution file.
        cov_path : :py:class:`~str` | None
            The path to the noise covariance file.
        src_path : :py:class:`~str` | None
            The path to the source space file.
        lf_path : :py:class:`~str` | None
            The path to the leadfield matrix file.
        save_all : :py:class:`~bool`
            If True, save the data and the forward model. Default to False.
        """
        write_pkl(fpath, self, sbj=sbj, sbj_viz=sbj_viz, data_path=data_path,
                  fwd_path=fwd_path, src_path=src_path, lf_path=lf_path,
                  save_all=save_all)
