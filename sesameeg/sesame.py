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
from mne.forward import _select_orient_forward
from .emp_pdf import EmpPdf
from .particles import Particle
from .utils import compute_neighbours_matrix, initialize_radius, \
    compute_neighbours_probability_matrix, estimate_s_noise, estimate_s_q
from .utils import is_evoked, is_forward
from .io import write_h5, write_pkl
from .metrics import compute_goodness_of_fit, compute_sd


class Sesame(object):
    """Sequential Semi-Analytic Monte-Carlo Estimation (SESAME) of sources.

    Parameters
    ----------
    forward : instance of Forward
        The forward solution.
    evoked : instance of mne.evoked.Evoked, EvokedArray
        The  data.
    s_noise : float
        The standard deviation of the noise distribution.
    radius : float | None
        The maximum distance in cm between two neighbouring verteces
        of the brain discretization. If None, radius = 1cm.
    sigma_neigh: float | None
        The standard deviation of the probability distribution of
        neighbours. If None sigma_neigh = radius/2.
    n_parts : int
        The number of particles forming the empirical pdf.
    sample_min : float | None
        First sample of the time window in which data are analyzed.
        If None, time window starts from the first sample of the data.
    sample_max : float | None
        Last sample of the time window in which dara are analyzed.
        If None, time window ends with the last sample of the data.
    subsample : int | None
        The step used to subsample the data. If None no subsampling is
        performed.
    s_q : float | None
        The standard deviation of the prior of the dipole moment.
        If None s_q is automatic estimated.
    cov : istance of Covariance | None
        The noise covariance matrix used to prewhiten the data. If None
        no prewhitening is applied.
    lam : float
        The parameter of the prior Poisson pdf of the number of dipoles.
    N_dip_max : int
        The maximum number of dipoles allowed in a particle.

    Attributes
    ----------
    n_parts : int
        The number of particles forming the empirical pdf.
    lam : float
        The parameter of the prior Poisson pdf of the number of dipoles.
    N_dip_max : int
        The maximum number of dipoles allowed in a particle.
    forward : instance of Forward
        The forward solution.
    source_space : array of floats, shape (n_verts, 3)
        The coordinates of the points in the brain discretization.
    n_verts : int
        The number of points forming the brain discretization.
    lead_field : array of floats, shape (n_sens x 3*n_verts)
        The leadfield matrix.
    distance_matrix : array of floats, shape (n_verts x n_verts)
        The euclidean distance matrix between the points in the
        brain discretization.
    neigh : array of ints, shape (n_vert, n_max_neigh)
        The set of neighbours of each point in the brain discretization.
        n_max_neigh is the cardinality of the biggest set.
    radius : float
        The radius used to compute the neigh matrix.
    neigh_p : array of floats, shape (n_vert, n_max_neigh)
        The neighbours' probabilities.
    sigma_neigh : float
        The standard deviation used to compute the neigh_p matrix.
    s_min : int
        The first sample of the time window in which data are analyzed.
    s_max : int
        The last sample of the time window in which data are analyzed.
    subsample : int | None
        The step used to subsample the data.
    r_data : array of floats, shape (n_sens, n_ist)
        The real part of the data; n_sens is the number of sensors and
        n_ist is the number of time-points or of frequencies.
    i_data : array of floats, shape (n_sens, n_ist)
        The imaginary part of the data; n_sens is the number of sensors
        and n_ist is the number of time-points or of frequencies.
    s_q : float
        The standard deviation of the prior of the dipole moment.
    hyper_q : bool
        If True use hyperprior in dipole strength
    s_noise : float
        The standard deviation of the noise distribution.
    _resample_it : list of ints
        The iterations during which a resampling step has been performed
    est_n_dips : list of ints
        The estimated number of dipoles for the first and the last iteration.
    est_locs : list of array of ints
        The estimated source locations for the first and the last iteration.
    est_q : None | array of floats, shape (n_ist x (3*est_n_dips[-1]))
        The sources' moments estimated at the last iteration.
        If None Sesame did not estimate the sources' moments yet.
    model_sel : list of arrays of floats
        The model selection (i.e. the posterior distribution of the number
        of dipoles) for the first and the last iteration.
    blob : list of 2D arrays of floats
        The intensity measure of the point process over the iterations.
    emp : instance of EmpPdf
        The empirical pdf approximated by the particles at each iteration.
    """

    def __init__(self, forward, evoked, s_noise=None, radius=None,
                 sigma_neigh=None, n_parts=100, sample_min=None,
                 sample_max=None, subsample=None, s_q=None, hyper_q=False,
                 cov=None, lam=0.25, N_dip_max=10, verbose=False):

        if not is_forward(forward):
            raise ValueError('Forward must be an instance of MNE'
                             ' class Forward.')

        if not is_evoked(evoked):
            raise ValueError('Data must be an instance of MNE class '
                             'Evoked or EvokedArray.')

        # 1) Choosen by the user
        self.n_parts = n_parts
        self.lam = lam
        self.N_dip_max = N_dip_max
        self.verbose = verbose
        self.hyper_q = hyper_q
        self.forward, _info_picked = _select_orient_forward(forward,
                                                            evoked.info, cov)

        self.source_space = forward['source_rr']
        self.n_verts = self.source_space.shape[0]
        self.lead_field = forward['sol']['data']

        self.distance_matrix = ssd.cdist(self.source_space, self.source_space)
        if radius is None:
            self.radius = initialize_radius(self.source_space)
        else:
            self.radius = radius
        print('Computing neighbours matrix...')
        self.neigh = compute_neighbours_matrix(self.source_space,
                                               self.distance_matrix,
                                               self.radius)
        print('[done]')

        if sigma_neigh is None:
            self.sigma_neigh = self.radius/2
        else:
            self.sigma_neigh = sigma_neigh
        print('Computing neighbours probabilities...')
        self.neigh_p = compute_neighbours_probability_matrix(self.neigh,
                                                             self.source_space,
                                                             self.distance_matrix,
                                                             self.sigma_neigh)
        print('[done]')

        if sample_min is None:
            self.s_min = 0
        else:
            if isinstance(sample_min, (int, np.integer)):
                self.s_min = sample_min
            else:
                raise ValueError('sample_min index should be an integer')

        if sample_max is None:
            self.s_max = evoked.data.shape[1]-1
        else:
            if isinstance(sample_max, (int, np.integer)):
                self.s_max = sample_max
            else:
                raise ValueError('sample_max index should be an integer')
        print('Analyzing data from {0} s to {1} s'.
              format(round(evoked.times[self.s_min], 4),
                     round(evoked.times[self.s_max], 4)))

        self.subsample = subsample

        if subsample is not None:
            print('Subsampling data with step {0}'.format(subsample))
            _data = evoked.data[:, self.s_min:self.s_max + 1:subsample]
        else:
            _data = evoked.data[:, self.s_min:self.s_max+1]

        # Perform whitening if a noise covariance is provided
        if cov is not None:
            whitener, _ = compute_whitener(cov, info=_info_picked, pca=True,
                                           picks=_info_picked['ch_names'])
            _data = np.sqrt(evoked.nave) * np.dot(whitener, _data)
            self.lead_field = (np.sqrt(evoked.nave) *
                               np.dot(whitener, self.lead_field))

        self.r_data = _data.real
        self.i_data = _data.imag
        del _data

        if s_q is None:
            print('Estimating dipole strength variance...')
            self.s_q = estimate_s_q(self.r_data, self.lead_field)
            print('[done]')
            print(' Estimated dipole strength variance: {:.4e}'
                  .format(self.s_q))
        elif isinstance(s_q, float):
            self.s_q = s_q
            print('User defined dipole strength variance: {:.4e}'
                  .format(self.s_q))
        else:
            raise ValueError('Sigma q must be either None or a float.')

        if self.hyper_q:
            print('Sampling hyperprior for dipole strength...')

        if s_noise is None:
            print('Estimating noise variance...')
            self.s_noise = estimate_s_noise(self.r_data)
            print('[done]')
            print(' Estimated noise variance: {:.4e}'.format(self.s_noise))
        else:
            self.s_noise = s_noise

        self._resample_it = list()
        self.est_n_dips = list()
        self.est_locs = list()
        self.est_q = None
        self.est_s_q = None
        self.final_s_q = None
        self.model_sel = list()
        self.blob = list()

        if self.hyper_q:
            self.est_s_q = list(np.array([]) for _ in range(self.n_parts))

        self.emp = EmpPdf(self.n_parts, self.n_verts, self.lam, s_q=self.s_q,
                          hyper_q=self.hyper_q, verbose=self.verbose)

        for _part in self.emp.particles:
            if self.hyper_q:
                _aux = 0
                _pos_def = _part._check_sigma(self.r_data, self.lead_field,
                                              self.s_noise)
                while _pos_def is False:
                    _part.s_q = 10 ** (3 * np.random.rand()) * (self.s_q / 35)
                    _pos_def = _part._check_sigma(self.r_data, self.lead_field,
                                                  self.s_noise)
                    _aux += 1
                    if _aux == 100:
                        raise ValueError

            _part.compute_loglikelihood_unit(self.r_data, self.lead_field,
                                             s_noise=self.s_noise)

    def apply_sesame(self, estimate_all=False, estimate_q=True):
        """Run SESAME and compute point estimates.

        Parameters
        ----------
        estimate_all : bool
            If True estimate the number of dipoles and their locations at
            each iteration. If False compute point-estimate only at the last
            iteration.
        estimate_q : bool
            If True compute point-estimate of the dipole moment at the
            last iteration.
        """

        print('Computing inverse solution. This will take a while...')
        # --------- INIZIALIZATION ------------
        # Samples are drawn from the prior distribution and weigths are set as
        # uniform.
        nd = np.array([_part.n_dips for _part in self.emp.particles])

        while not np.all(nd <= self.N_dip_max):
            nd_wrong = np.where(nd > self.N_dip_max)[0]
            self.emp.particles[nd_wrong] =\
                np.array([Particle(self.n_verts, self.lam, s_q=self.s_q)
                         for _ in itertools.repeat(None, nd_wrong.shape[0])])
            nd = np.array([_part.n_dips for _part in self.emp.particles])

        # Point estimation for the first iteration
        if self.hyper_q:
            for i_p, _part in enumerate(self.emp.particles):
                self.est_s_q[i_p] = np.append(self.est_s_q[i_p],
                                              _part.s_q)
        if estimate_all:
            self.emp.point_estimate(self.distance_matrix, self.N_dip_max)

            self.est_n_dips.append(self.emp.est_n_dips)
            self.model_sel.append(self.emp.model_sel)
            self.est_locs.append(self.emp.est_locs)
            self.blob.append(self.emp.blob)

        # ----------- MAIN CICLE --------------

        while np.all(self.emp.exponents <= 1):
            time_start = time.time()
            if self.verbose:
                print('iteration = {0}'.format(self.emp.exponents.shape[0]))
                print('exponent = {0}'.format(self.emp.exponents[-1]))
                print('ESS = {:.2%}'.format(self.emp.ESS/self.n_parts))

            # STEP 1: (possible) resampling
            if self.emp.ESS < self.n_parts/2:
                self._resample_it.append(int(self.emp.exponents.shape[0]))
                self.emp.resample()
                if self.verbose:
                    print('----- RESAMPLING -----')
                    print('ESS = {:.2%}'.format(self.emp.ESS/self.n_parts))

            # STEP 2: Sampling.
            self.emp.sample(self.n_verts, self.r_data, self.lead_field,
                            self.neigh, self.neigh_p, self.s_noise,
                            self.lam, self.N_dip_max)

            # STEP 3: Point Estimation
            if self.hyper_q:
                for i_p, _part in enumerate(self.emp.particles):
                    self.est_s_q[i_p] = np.append(self.est_s_q[i_p],
                                                  _part.s_q)
            if estimate_all:
                self.emp.point_estimate(self.distance_matrix, self.N_dip_max)

                self.est_n_dips.append(self.emp.est_n_dips)
                self.model_sel.append(self.emp.model_sel)
                self.est_locs.append(self.emp.est_locs)
                self.blob.append(self.emp.blob)

            # STEP 4: compute new exponent and new weights
            self.emp.compute_exponent(self.s_noise)

            time.sleep(0.01)
            time_elapsed = (time.time() - time_start)
            if self.verbose:
                print('Computation time: {:.2f} seconds'.format(time_elapsed))
                print('-------------------------------')

        # Point estimation
        if self.hyper_q:
            for i_p, _part in enumerate(self.emp.particles):
                self.est_s_q[i_p] = np.append(self.est_s_q[i_p],
                                              _part.s_q)
        self.emp.point_estimate(self.distance_matrix, self.N_dip_max)

        self.est_n_dips.append(self.emp.est_n_dips)
        self.model_sel.append(self.emp.model_sel)
        self.est_locs.append(self.emp.est_locs)
        self.blob.append(self.emp.blob)
        if estimate_q:
            if self.est_n_dips[-1] == 0:
                self.est_q = np.array([])
            else:
                self.compute_q(self.est_locs[-1])
        print('[done]')

    def compute_q(self, est_locs):
        """Compute a point estimate for the dipole moments.

        Parameters
        ----------
        est_locs : list of array of ints
            The estimated source locations.
        """
        est_num = est_locs.shape[0]
        [n_sens, n_time] = np.shape(self.r_data)

        if self.hyper_q:
            weights = np.exp(self.emp.logweights)
            assert np.abs(np.sum(weights) - 1) < 1e-15
            est_sq = np.asarray([self.est_s_q[p][-1] for p in range(self.n_parts)])
            _s_q = np.dot(weights, est_sq)
            self.final_s_q = _s_q
            print('Estimated dipole strength variance: {}'.format(_s_q))
        else:
            _s_q = self.s_q

        ind = np.ravel([[3*est_locs[idip], 3*est_locs[idip]+1,
                       3*est_locs[idip]+2] for idip in range(est_num)])
        Gc = self.lead_field[:, ind]
        sigma = (_s_q / self.s_noise)**2 * np.dot(Gc, np.transpose(Gc)) +\
            np.eye(n_sens)
        kal_mat = (_s_q / self.s_noise)**2 * np.dot(np.transpose(Gc),
                                                        np.linalg.inv(sigma))
        self.est_q = np.array([np.dot(kal_mat, self.r_data[:, t])
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
        gof : float
            The goodness of fit with the recorded data.
        """

        if len(self.est_n_dips) == 0:
            raise AttributeError("No estimation found."
                                 "Run apply_sesame first.")
        if self.est_q is None:
            raise AttributeError("No dipoles' moment found."
                                 " Run compute_q first.")

        gof = compute_goodness_of_fit(self.r_data, self.est_n_dips[-1],
                                      self.est_locs[-1], self.est_q,
                                      self.lead_field)
        return gof

    def source_dispersion(self):
        blob_tot = np.sum(self.blob[-1], axis=0)
        est_pos = self.source_space[self.est_locs[-1]]
        sd = compute_sd(self.source_space, blob_tot, est_pos)
        return sd

    def compute_stc(self, subject=None):
        """Compute a SourceEstimate object from the posterior pdf
        :math:`p(r|\\mathbf{y}, \\hat{n}_D)`, where :math:`\\hat{n}_D`
        is the estimated number of sources.
        For each point :math:`r` of the brain discretization
        :math:`p(r|\\mathbf{y}, \\hat{n}_D)` is the probability of a
        source being located in :math:`r`.

        Parameters
        ----------
        subject : str | None
            The subject name.

        Returns
        --------
        stc : instance of SourceEstimate.
            The source estimate object containing the posterior map of the
            dipoles' location.
        """
        if self.forward['src'].kind == 'surface':
            print('Surface stc computed.')
            return self.to_stc(subject=subject)
        elif self.forward['src'].kind == 'volume':
            raise NotImplementedError
            # print('Volume stc computed  ')
            # return self.to_vol_stc(subject=subject)
        else:
            raise ValueError('src can be either surface or volume')

    def to_stc(self, subject=None):
        """Compute a SourceEstimate object from the posterior pdf
        :math:`p(r|\\mathbf{y}, \\hat{n}_D)`, where :math:`\\hat{n}_D`
        is the estimated number of sources.
        For each point :math:`r` of the brain discretization
        :math:`p(r|\\mathbf{y}, \\hat{n}_D)` is the probability of a
        source being located in :math:`r`.

        Parameters
        ----------
        subject : str | None
            The subject name.

        Returns
        --------
        stc : instance of SourceEstimate.
            The source estimate object containing the posterior map of the
            dipoles' location.
        """

        if 'SourceEstimate' not in dir():
            from mne import SourceEstimate

        if not hasattr(self, 'blob'):
            raise AttributeError('Run SESAME first!!')

        blobs = self.blob
        fwd = self.forward
        est_n_dips = self.est_n_dips
        vertno = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
        nv_tot = fwd['nsource']

        blob_tot = np.zeros([len(blobs), nv_tot])
        for it, bl in enumerate(blobs):
            if est_n_dips[it] > 0:
                blob_tot[it] = np.sum(bl, axis=0)

        tmin = 1
        stc = SourceEstimate(data=blob_tot.T, vertices=vertno, tmin=tmin,
                             tstep=1, subject=subject)
        return stc

    def save_h5(self, fpath, tmin=None, tmax=None, subsample=None,
                sbj=None, sbj_viz=None, data_path=None, fwd_path=None,
                src_path=None, lf_path=None):
        """Save SESAME result to an HDF5 file.

        Parameters
        ----------
        fpath: str
            The path to the save file
        tmin: float | None
            The first instant (in seconds) of the time window in which data have been analyzed.
        tmax: float | None
            The last instant (in seconds) of the time window in which data have been analyzed.
        subsample : int | None
            The step used to subsample the data.
        sbj : str | None
            The subject name.
        sbj_viz: str | None
            The name of the subject's FreeSurfer folder.
        data_path: str | None
            The path to the data file
        fwd_path: str | None
            The path to the forward solution file
        src_path: str | None
            The path to the source space file
        lf_path: str | None
            The path to the leadfield matrix file
        """
        write_h5(fpath, self, tmin=tmin, tmax=tmax, subsample=subsample,
                 sbj=sbj, sbj_viz=sbj_viz, data_path=data_path,
                 fwd_path=fwd_path, src_path=src_path, lf_path=lf_path)

    def save_pkl(self, fpath, tmin=None, tmax=None, subsample=None,
                sbj=None, sbj_viz=None, data_path=None, fwd_path=None,
                src_path=None, lf_path=None):
        """Save SESAME result to an Python pickle file.

        Parameters
        ----------
        fpath: str
            The path to the save file
        tmin: float | None
            The first instant (in seconds) of the time window in which data have been analyzed.
        tmax: float | None
            The last instant (in seconds) of the time window in which data have been analyzed.
        subsample : int | None
            The step used to subsample the data.
        sbj : str | None
            The subject name.
        sbj_viz: str | None
            The name of the subject's FreeSurfer folder.
        data_path: str | None
            The path to the data file
        fwd_path: str | None
            The path to the forward solution file
        src_path: str | None
            The path to the source space file
        lf_path: str | None
            The path to the leadfield matrix file
        """
        write_pkl(fpath, self, tmin=tmin, tmax=tmax, subsample=subsample,
                  sbj=sbj, sbj_viz=sbj_viz, data_path=data_path,
                  fwd_path=fwd_path, src_path=src_path, lf_path=lf_path)