# -*- coding: utf-8 -*-

# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#          Alberto Sorrentino <sorrentino@dima.unige.it>
#
# License: BSD (3-clause)

import itertools
import time
import numpy as np
import scipy.spatial.distance as ssd
from mne.cov import compute_whitener
from mne.evoked import EvokedArray
from mne.forward import _select_orient_forward
from .emp_pdf import EmpPdf
from .particles import Particle
from .utils import compute_neighbours_matrix, initialize_radius, \
    compute_neighbours_probability_matrix, estimate_noise_std, estimate_dip_mom_std
from .utils import is_epochs, is_evoked, is_forward
from .io import _export_to_stc, _export_to_vol_stc, write_h5, write_pkl
from .metrics import compute_goodness_of_fit, compute_sd


class Sesame(object):
    """Sequential Semi-Analytic Monte-Carlo Estimation (SESAME) of sources.

    Parameters
    ----------
    forward : :py:class:`~mne.Forward` object
        The forward solution.
    data : instance of :py:class:`~mne.Evoked` | :py:class:`~mne.EvokedArray`
        The MEEG data.
    noise_std : :py:class:`~float` | None
        The standard deviation of the noise distribution.
        If None, it is estimated from the data
    radius : :py:class:`~float` | None
        The maximum distance in cm between two neighbouring vertices
        of the brain discretization. If None, radius is set to 1 cm.
    neigh_std : :py:class:`~float` | None
        The standard deviation of the probability distribution of
        neighbours. If None, neighb_std is set to radius/2.
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
    dip_mom_std : :py:class:`~float` | None
        The standard deviation of the prior pdf on the dipole moment.
        If None, it is estimated from the forward model and the data.
    hyper_q : :py:class:`~bool`
        If True an hyperprior pdf on the dipole moment std will be used.
        Default is True.
    noise_cov : instance of :py:class:`~mne.Covariance` | None
        The noise covariance matrix used to prewhiten the data. If None,
        no prewhitening is applied.
    lam : :py:class:`~float`
        The parameter of the Poisson prior pdf on the number of dipoles.
    max_n_dips : :py:class:`~int`
        The maximum number of dipoles allowed in a particle.
    Fourier_transf : :py:class:`~bool`
        If True data are converted to the frequency domain.
    verbose : :py:class:`~bool`
        If True, increase verbose level.

    Attributes
    ----------
    n_parts : :py:class:`~int`
        The number of particles forming the empirical pdf.
    lam : :py:class:`~float`
        The parameter of the Poisson prior pdf on the number of dipoles.
    max_n_dips : :py:class:`~int`
        The maximum number of dipoles allowed in a particle.
    forward : instance of :py:class:`~mne.Forward`
        The forward solution.
    source_space : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts, 3)
        The coordinates of the points in the brain discretization.
    n_verts : :py:class:`~int`
        The number of points forming the brain discretization.
    lead_field : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_sens x 3*n_verts)
        The leadfield matrix.
    distance_matrix : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts x n_verts)
        The Euclidean distance between the points in the
        brain discretization.
    neigh : :py:class:`~numpy.ndarray` of :py:class:`~int`, shape (n_vert, n_max_neigh)
        The set of neighbours of each point in the brain discretization.
        n_max_neigh is the cardinality of the biggest set.
    radius : :py:class:`~float`
        The radius used to compute the neighbours.
    neigh_p : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_vert, n_max_neigh)
        The neighbours' probabilities.
    neigh_std : :py:class:`~float`
        The standard deviation used to compute the neigh_p matrix.
    s_min : :py:class:`~int`
        The first sample of the segment of data that are analyzed.
    s_max : :py:class:`~int`
        The last sample of the segment of data that are analyzed.
    subsample : :py:class:`~int` | None
        The step used to subsample the data.
    r_data : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_sens, n_ist)
        The real part of the data; n_sens is the number of sensors and
        n_ist is the number of time-points or of frequencies.
    dip_mom_std : :py:class:`~float`
        The standard deviation of the prior on the dipole moment.
    hyper_q : :py:class:`~bool`
        If True use hyperprior on dipole moment std.
    noise_std : :py:class:`~float`
        The standard deviation of the noise distribution.
    _resample_it : :py:class:`~list` of :py:class:`~int`
        The iterations during which a resampling step has been performed
    est_n_dips : :py:class:`~list` of :py:class:`~int`
        The estimated number of dipoles.
    est_locs : :py:class:`~list` of :py:class:`~numpy.ndarray` of :py:class:`~int`
        The source space grid points indices in which a source is estimated.
    est_dip_moms : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_ist x (3*est_n_dips[-1])) | None
        The sources' moments estimated at the last iteration.
        If None, moments can be estimated by calling :py:meth:`~Sesame.compute_dip_mom`
    model_sel : :py:class:`~list` of :py:class:`~numpy.ndarray` of :py:class:`~float`
        The model selection, i.e. the posterior distribution on the number
        of dipoles.
    pmap : :py:class:`~list` of :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (est_n_dips, n_verts)
        Posterior probability map.
    posterior : instance of :class:`~sesameeg.emppdf.EmpPdf`
        The empirical pdf approximated by the particles at each iteration.
    """

    def __init__(self, forward, data, noise_std=None, radius=None,
                 neigh_std=None, n_parts=100, top_min=None,
                 top_max=None, subsample=None, dip_mom_std=None, hyper_q=True,
                 noise_cov=None, lam=0.25, max_n_dips=10, Fourier_transf=False,
                 verbose=False):

        if not is_forward(forward):
            raise ValueError('Forward must be an instance of MNE'
                             ' class Forward.')

        if not is_evoked(data):
            raise ValueError('Data must be an instance of MNE class '
                             'Evoked or EvokedArray.')

        # 1) Choosen by the user
        self.fourier = Fourier_transf
        self.n_parts = n_parts
        self.lam = lam
        self.max_n_dips = max_n_dips
        self.subsample = subsample
        self.verbose = verbose
        self.hyper_q = hyper_q
        self.forward, _info_picked = _select_orient_forward(forward,
                                                            data.info, noise_cov)

        self.source_space = forward['source_rr']
        self.n_verts = self.source_space.shape[0]
        self.lead_field = forward['sol']['data']

        self.distance_matrix = ssd.cdist(self.source_space, self.source_space)
        if radius is None:
            self.radius = initialize_radius(self.source_space)
        else:
            self.radius = radius
        print('Computing neighbours matrix...', end='')
        self.neigh = compute_neighbours_matrix(self.source_space,
                                               self.distance_matrix,
                                               self.radius)
        print('[done]')

        if neigh_std is None:
            self.neigh_std = self.radius/2
        else:
            self.neigh_std = neigh_std
        print('Computing neighbours probabilities...', end='')
        self.neigh_p = compute_neighbours_probability_matrix(self.neigh,
                                                             self.source_space,
                                                             self.distance_matrix,
                                                             self.neigh_std)
        print('[done]')

        # Prepare data
        if is_evoked(data):
            _data = self._prepare_evoked_data(data, top_min, top_max)
        else:
            raise ValueError

        # Perform whitening if a noise covariance is provided
        if noise_cov is not None:
            if self.fourier:
                raise NotImplementedError('Still to implement whitening in the frequency domain')
            else:
                if is_evoked(data):
                    _sf = np.sqrt(data.nave)
                elif is_epochs(data):
                    _sf = 1.0
                else:
                    raise ValueError

                whitener, _ = compute_whitener(noise_cov, info=_info_picked, pca=True,
                                               picks=_info_picked['ch_names'])
                _data = _sf * np.dot(whitener, _data)
                self.lead_field = (_sf * np.dot(whitener, self.lead_field))
                print('Data and leadfield have been prewhitened.')

        self.r_data = _data.real
        del _data

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

    def _prepare_epochs_data(self, epochs, top_min, top_max):
        ep_data = epochs.get_data()
        evoked = EvokedArray(ep_data[0], epochs.info)
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
            for ie, _e in enumerate(ep_data):
                _e *= np.hamming(_e.shape[1])
                _e_f = np.fft.rfft(_e)
                if self.subsample is not None:
                    if ie == 0:
                        print('Subsampling data with step {0}'.format(self.subsample))
                    temp_list.append(_e_f[:, self.s_min:self.s_max + 1:self.subsample])
                else:
                    temp_list.append(_e_f[:, self.s_min:self.s_max + 1])

                for _data_temp in temp_list:
                    for l in _data_temp.T:
                        temp_list2.append(np.vstack([np.real(l), np.imag(l)]).T)
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
            for l in _data_temp.T:
                temp_list.append(np.vstack([np.real(l), np.imag(l)]).T)
            return np.hstack(temp_list)
        else:
            raise ValueError

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

        if self.posterior.exponents[-1] >0:
            print('Resetting SESAME...', end='')
            self._reset_attributes()
            print('[done]')

        print('Computing inverse solution. This will take a while...')
        # --------- INIZIALIZATION ------------
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
        if self.hyper_q:
            for i_p, _part in enumerate(self.posterior.particles):
                self.est_dip_mom_std[i_p] = np.append(self.est_dip_mom_std[i_p],
                                              _part.dip_mom_std)
        if estimate_all:
            self.posterior.point_estimate(self.distance_matrix, self.max_n_dips)

            self.est_n_dips.append(self.posterior.est_n_dips)
            self.model_sel.append(self.posterior.model_sel)
            self.est_locs.append(self.posterior.est_locs)
            self.pmap.append(self.posterior.pmap)

        # ----------- MAIN CICLE --------------

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
            if self.hyper_q:
                for i_p, _part in enumerate(self.posterior.particles):
                    self.est_dip_mom_std[i_p] = np.append(self.est_dip_mom_std[i_p],
                                                  _part.dip_mom_std)
            if estimate_all:
                self.posterior.point_estimate(self.distance_matrix, self.max_n_dips)

                self.est_n_dips.append(self.posterior.est_n_dips)
                self.model_sel.append(self.posterior.model_sel)
                self.est_locs.append(self.posterior.est_locs)
                self.pmap.append(self.posterior.pmap)

            # STEP 4: compute new exponent and new weights
            self.posterior.compute_exponent(self.noise_std)

            time.sleep(0.01)
            time_elapsed = (time.time() - time_start)
            if self.verbose:
                print('Computation time: {:.2f} seconds'.format(time_elapsed))
                print('-------------------------------')

        # Point estimation
        if self.hyper_q:
            for i_p, _part in enumerate(self.posterior.particles):
                self.est_dip_mom_std[i_p] = np.append(self.est_dip_mom_std[i_p],
                                              _part.dip_mom_std)
        self.posterior.point_estimate(self.distance_matrix, self.max_n_dips)

        self.est_n_dips.append(self.posterior.est_n_dips)
        self.model_sel.append(self.posterior.model_sel)
        self.est_locs.append(self.posterior.est_locs)
        self.pmap.append(self.posterior.pmap)
        if estimate_dip_mom:
            if self.est_n_dips[-1] == 0:
                self.est_dip_moms = np.array([])
                if self.hyper_q:
                    weights = np.exp(self.posterior.logweights)
                    assert np.abs(np.sum(weights) - 1) < 1e-15
                    est_sq = np.asarray([self.est_dip_mom_std[p][-1] for p in range(self.n_parts)])
                    self.final_dip_mom_std = np.dot(weights, est_sq)
                    print('Estimated dipole strength variance: {}'.format(self.final_dip_mom_std))
            else:
                self.compute_dip_mom(self.est_locs[-1])
        print('[done]')

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
            weights = np.exp(self.posterior.logweights)
            assert np.abs(np.sum(weights) - 1) < 1e-15
            est_sq = np.asarray([self.est_dip_mom_std[p][-1] for p in range(self.n_parts)])
            _dip_mom_std = np.dot(weights, est_sq)
            self.final_dip_mom_std = _dip_mom_std
            print('Estimated dipole strength variance: {}'.format(_dip_mom_std))
        else:
            _dip_mom_std = self.dip_mom_std

        ind = np.ravel([[3*est_locs[idip], 3*est_locs[idip]+1,
                       3*est_locs[idip]+2] for idip in range(est_num)])
        Gc = self.lead_field[:, ind]
        sigma = (_dip_mom_std / self.noise_std)**2 * np.dot(Gc, np.transpose(Gc)) +\
            np.eye(n_sens)
        kal_mat = (_dip_mom_std / self.noise_std)**2 * np.dot(np.transpose(Gc),
                                                        np.linalg.inv(sigma))
        self.est_dip_moms = np.array([np.dot(kal_mat, self.r_data[:, t])
                              for t in range(n_time)])

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

    def compute_stc(self, subject=None):
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
            --------
            stc : :py:class:`~mne.SourceEstimate` | :py:class:`~mne.VolSourceEstimate`
                The source estimate object containing the posterior map of the
                dipoles' location.
            """
        if self.forward['src'].kind == 'surface':
            print('Surface stc computed.')
            return _export_to_stc(self, subject=subject)
        elif self.forward['src'].kind == 'volume':
            print('Volume stc computed  ')
            return _export_to_vol_stc(self, subject=subject)
        else:
            raise ValueError('src can be either surface or volume')

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
        if self.fourier is False:
            write_h5(fpath, self, tmin=self.top_min, tmax=self.top_max,
                     subsample=self.subsample, sbj=sbj, sbj_viz=sbj_viz,
                     data_path=data_path, fwd_path=fwd_path, src_path=src_path,
                     lf_path=lf_path)
        else:
            write_h5(fpath, self, fmin=self.top_min, fmax=self.top_max,
                     subsample=self.subsample, sbj=sbj, sbj_viz=sbj_viz,
                     data_path=data_path, fwd_path=fwd_path, src_path=src_path,
                     lf_path=lf_path)

    def save_pkl(self, fpath, sbj=None, sbj_viz=None, data_path=None,
                 fwd_path=None, cov_path=None, src_path=None,
                 lf_path=None, save_all=False):
        """Save SESAME result to an Python pickle file.

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
