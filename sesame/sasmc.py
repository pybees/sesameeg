# -*- coding: utf-8 -*-

# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#          Alberto Sorrentino <sorrentino@dima.unige.it>
#
# License: BSD (3-clause)

from __future__ import division
# TODO: check and import only if necessary

import numpy as np
import scipy.spatial.distance as ssd
import copy
import time
import itertools
from mne.cov import compute_whitener
from mne.forward import _select_orient_forward
import mne
from mayavi import mlab


def estimate_noise_std(evoked, sample_min=None, sample_max=None, picks=None):
    '''Estimate the standard deviation of the noise distribution from a
    portion of the data which is assumed to be generated from noise only.

    Parameters
    ----------
    evoked: Evoked object
        Evoked data
    sample_min : float | None
        First sample of the portion of the data used.
        If None, it will be set equal to the first sample of the data.
    sample_max : float | None
        Last sample of the portion of the data used.
        If None, it will be set equal to the last sample of the data.
    picks: array-like of int | None
        indices of selected channels. If None, all channels are used.

    Returns
    -------
    s_noise : float
        Estimated noise standard deviation
    '''

    # TODO: gestire meglio i picks
    # (consentire una scrittura tipo picks = 'grad')

    _data = evoked.data

    if picks is None:
        prestimulus = _data[:, sample_min:sample_max + 1]
    else:
        prestimulus = _data[picks, sample_min:sample_max + 1]

    s_noise = np.mean(np.std(prestimulus, axis=1))
    return s_noise


class Dipole(object):
    """Single current dipole class for semi-analytic SMC algorithm

    Parameters
    ----------
    loc : int
        The dipole location (as an index of a brain grid).
    """

    def __init__(self, loc):
        self.loc = loc

    def __repr__(self):
        s = 'location : {0}'.format(self.loc)
        return '<Dipole  |  {0}>'.format(s)

    # TODO: decidere se aggiungere momento di dipolo 
    # e (forse) coordinate del vertice


class Particle(object):
    """Particle class for the semi-analytic SMC algorithm.
    Used to store a single particle of an empirical pdf.

    Parameters
    ----------
    n_verts : int
        The number of the points in the given brain discretization.
    lam : float
        The parameter of the prior Poisson pdf of the number of dipoles.

    Attributes
    ----------
    n_dips : int
        The number of dipoles in the particle.
    dipoles : array of instances of Dipole, shape(n_dips,)
        The particle's dipoles.
    loglikelihood_unit : float
        The logarithm of the marginal likelihood, evaluated in the particle.
    prior : float
        The prior pdf, evaluated in the particle.
    """

    def __init__(self, n_verts, lam):
        """Initialization: the initial number of dipoles is Poisson
           distribuited; the initial locations are uniformly distribuited
           within the brain grid points, with no dipoles in the same position.
        """
        self.n_dips = 0
        self.dipoles = np.array([])
        self.prior = None
        self.loglikelihood_unit = None

        self.add_dipole(n_verts, np.random.poisson(lam))
        self.compute_prior(lam)

    def __repr__(self):
        s = 'n_dips : {0}'.format(self.n_dips)
        for i_dip, dip in enumerate(self.dipoles):
            s += ', dipole {0} : {1}' .format(i_dip+1, dip)
        s += ', prior : {0}'.format(self.prior)
        return '<Particle  |  {0}>'.format(s)

    def add_dipole(self, n_verts, num_dip=1):
        """ Add new dipoles to the particle.

        Parameters
        ----------
        n_verts : int
            The number of the points in the given brain discretization.
        num_dip : int
            The number of dipoles to add.
        """

        new_locs = np.random.randint(0, n_verts, num_dip)

        for loc in new_locs:
            while loc in [dip.loc for dip in self.dipoles]:
                loc = np.random.randint(0, n_verts)

            self.dipoles = np.append(self.dipoles, Dipole(loc))
            self.n_dips += 1

    def remove_dipole(self, diprip):
        """ Remove a dipole from the particle.

        Parameters
        ----------
        diprip : int
            The index representing the dipoles array entry to be removed.
        """

        if self.dipoles.shape[0] > 0:
            self.dipoles = np.delete(self.dipoles, diprip)
            self.n_dips -= 1
        else:
            raise ValueError('No dipoles to remove.')

    def compute_loglikelihood_unit(self, r_data, lead_field, s_noise, s_q):
        """ Evaluates the logarithm of the likelihood function in the present particle.

        Parameters
        ----------
        r_data : array of floats, shape (n_sens, n_ist)
            The real part of the data; n_sens is the number of sensors and
            n_ist is the number of time-points or of frequencies.
        lead_field : array of floats, shape (n_sens x 3*n_verts)
            The leadfield matrix.
        s_noise : float
            The standard deviation of the noise distribution.
        s_q : float
            The standard deviation of the prior pdf of the dipole moment.

        Returns
        -------
        loglikelihood_unit : float
            The logarithm of the likelihood function in the present particle.
        """

        [n_sens, n_ist] = r_data.shape

        # Step 1: compute variance of the likelihood.
        if self.n_dips == 0:
            sigma = np.eye(n_sens)
        else:
            # 1a: compute the leadfield of the particle
            idx = np.ravel([[3*dip.loc, 3*dip.loc+1, 3*dip.loc+2] for dip in self.dipoles])
            Gc = lead_field[:, idx]
            # 1b: compute the variance
            sigma = (s_q / s_noise)**2 * np.dot(Gc, np.transpose(Gc)) + \
                np.eye(n_sens)

        # Step 2: compute inverse and determinant of the variance
        inv_sigma = np.linalg.inv(sigma)
        det_sigma = np.linalg.det(sigma)

        # Step 3: compute the log_likelihood
        self.loglikelihood_unit = - (n_ist * s_noise**2) * np.log(det_sigma)
        for ist in range(n_ist):
            self.loglikelihood_unit -= \
                np.transpose(r_data[:, ist]).dot(inv_sigma).dot(r_data[:, ist])
        return self.loglikelihood_unit

    def compute_prior(self, lam):
        """Evaluate the prior pdf in the present particle.

        Parameters
        ----------
        lam : float
            Parameter of the Poisson probability distribution used for
            determining the number of dipoles in the particle.

        Returns
        -------
        prior : float
            The prior pdf evaluated in the present particle.
        """
        self.prior = 1/np.math.factorial(self.n_dips) * np.exp(-lam) *\
            (lam**self.n_dips)
        return self.prior

    def evol_n_dips(self, n_verts, r_data, lead_field, N_dip_max,
                    lklh_exp, s_noise, sigma_q, lam, q_birth=1 / 3,
                    q_death=1 / 20):
        """Perform a Reversible Jump Markov Chain Monte Carlo step in order
           to explore the "number of sources" component of the state space.
           Recall that we are working in a variable dimension model.

        Parameters
        ----------
        n_verts : int
            The number of the points in the given brain discretization.
        r_data : array of floats, shape (n_sens, n_ist)
            The real part of the data; n_sens is the number of sensors and
            n_ist is the number of time-points or of frequencies.
        lead_field : array of floats, shape (n_sens x 3*n_verts)
            The leadfield matrix.
        N_dip_max : int
            The maximum number of dipoles allowed in the particle.
        lklh_exp : float
            This number represents a point in the sequence of artificial
            distributions used in the SASMC samplers algorithm.
        s_noise : float
            The standard deviation of the noise distribution.
        s_q : float
            standard deviation of the prior of the dipole moment.
        lam : float
            Parameter of the Poisson probability distribution used for
            determining the number of dipoles in the particle.
        q_birth : ----
        q_death : ----

        Return
        ------
        self : instance of Particle
            The possibly modified particle instance.
        """

        prop_part = copy.deepcopy(self)
        birth_death = np.random.uniform(1e-16, 1)

        if self.loglikelihood_unit is None:
            self.compute_loglikelihood_unit(r_data, lead_field,
                                            s_noise, sigma_q)

        if birth_death < q_birth and prop_part.n_dips < N_dip_max:
            prop_part.add_dipole(n_verts)
        elif prop_part.n_dips > 0 and birth_death > 1-q_death:
            sent_to_death = np.random.random_integers(0, self.n_dips-1)
            prop_part.remove_dipole(sent_to_death)

        # Compute alpha rjmcmc
        if prop_part.n_dips != self.n_dips:
            prop_part.compute_prior(lam)
            prop_part.compute_loglikelihood_unit(r_data, lead_field,
                                                 s_noise, sigma_q)
            log_prod_like = prop_part.loglikelihood_unit - self.loglikelihood_unit

            if prop_part.n_dips > self.n_dips:
                alpha = np.amin([1, ((q_death * prop_part.prior) /
                                     (q_birth * self.prior)) * np.exp((lklh_exp/(2*s_noise**2)) * log_prod_like)])
            elif prop_part.n_dips < self.n_dips:
                alpha = np.amin([1, ((q_birth * prop_part.prior) /
                                     (q_death * self.prior)) * np.exp((lklh_exp/(2*s_noise**2)) * log_prod_like)])

            if np.random.rand() < alpha:
                self = copy.deepcopy(prop_part)
        return self

    def evol_loc(self, dip_idx, neigh, neigh_p, r_data, lead_field,
                 lklh_exp, s_noise, sigma_q, lam):
        """Perform a Markov Chain Monte Carlo step in order to explore the
           dipole location component of the state space. The dipole is
           allowed to move only to a restricted set of brain points,
           called "neighbours", with a probability that depends on the point.

        Parameters
        ----------
        dip_idx : int
            index of the Particle.dipoles array.
        neigh : array of ints
            The neighbours of each point in the brain discretization.
        neigh_p : array of floats
            The neighbours' probabilities.
        r_data : array of floats, shape (n_sens, n_ist)
            The real part of the data; n_sens is the number of sensors and
            n_ist is the number of time-points or of frequencies.
        lead_field : array of floats, shape (n_sens x 3*n_verts)
            The leadfield matrix.
        lklh_exp : float
            This number represents a point in the sequence of artificial
            distributions used in the SASMC samplers algorithm.
        s_noise : float
            The standard deviation of the noise distribution.
        sigma_q : float
            standard deviation of the prior of the dipole moment.
        lam : float
            Parameter of the Poisson probability distribution used for
            determining the number of dipoles in the particle.

        Return
        ------
        self : instance of Particle
            The possibly modified particle instance.
        """
        # Step 1: Drawn of the new location.
        prop_part = copy.deepcopy(self)
        p_part = np.cumsum(neigh_p[prop_part.dipoles[dip_idx].loc,
                           np.where(neigh[prop_part.dipoles[dip_idx].loc] != -1)])
        new_pos = False

        while new_pos is False:
            n_rand = np.random.random_sample(1)
            ind_p = np.digitize(n_rand, p_part)[0]
            prop_loc = neigh[prop_part.dipoles[dip_idx].loc, ind_p]
            new_pos = True

            for k in np.delete(range(prop_part.n_dips), dip_idx):
                if prop_loc == prop_part.dipoles[k].loc:
                    new_pos = False

        prob_new_move = neigh_p[prop_part.dipoles[dip_idx].loc, ind_p]

        prob_opp_move = neigh_p[prop_loc,
                                np.argwhere(neigh[prop_loc] ==
                                            prop_part.dipoles[dip_idx].loc)[0][0]]
        prop_part.dipoles[dip_idx].loc = prop_loc
        comp_fact_delta_r = prob_opp_move / prob_new_move

        # Compute alpha mcmc
        prop_part.compute_prior(lam)
        prop_part.compute_loglikelihood_unit(r_data, lead_field,
                                             s_noise, sigma_q)

        if self.loglikelihood_unit is None:
            self.compute_loglikelihood_unit(r_data, lead_field,
                                            s_noise, sigma_q)

        log_prod_like = prop_part.loglikelihood_unit - self.loglikelihood_unit
        alpha = np.amin([1, (comp_fact_delta_r *
                         (prop_part.prior/self.prior) *
                         np.exp((lklh_exp/(2*s_noise**2)) * log_prod_like))])

        if np.random.rand() < alpha:
            self = copy.deepcopy(prop_part)

        return self


class EmpPdf(object):
    """ Empirical probability density function class for the
        semi-analytic SMC samplers algorithm.

    Parameters
    ----------
    n_parts : int
        The number of particles forming the empirical pdf.
    n_verts : int
        The number of the points in the given brain discretization.
    lam : float
        The parameter of the prior Poisson pdf of the number of dipoles.

    Attributes
    ----------
    particles : array of instances of Particle, shape(n_parts,)
        The EmpPdf's particles.
    logweights : array of floats, shape(n_parts,)
        The logarithm of the weights of the particles forming the
        Empirical pdf.
    ESS : float
        The Effective Sample Size
    exponents : array of floats
        Array whose entries represent points in the space of artificial
        distributions. It is used to keep track of the path followed
        by the SASMC samplers algorithm.
    model_sel : array of floats
        Marginal posterior probability of the number of sources.
    est_n_dips : float
        Estimated number of sources.
    blob : array of floats, shape(est_n_dips x n_verts)
        Intensity measure of the point process.
    est_locs : array of ints
        Estimated sources locations
    """
    def __init__(self, n_parts, n_verts, lam, verbose=False):
        self.particles = np.array([Particle(n_verts, lam) for _ in itertools.repeat(None, n_parts)])
        self.logweights = np.array([np.log(1/n_parts) for _ in itertools.repeat(None, n_parts)])
        self.ESS = np.float32(1. / np.square(np.exp(self.logweights)).sum())
        self.exponents = np.array([0, 0])
        self.model_sel = None
        self.est_n_dips = None
        self.blob = None
        self.est_locs = None
        self.verbose = verbose
        # TODO: controlla di non aver fatto casino con le dichiarazioni

    def __repr__(self):
        s = ''
        for i_p, _part in enumerate(self.particles):
            s += '---- Particle {0} (W = {1},  number of dipoles = {2}): \n {3} \n'.format(i_p+1,
                                                                                           np.exp(self.logweights[i_p]),
                                                                                           _part.nu, _part)
        return s

    def sample(self, n_verts, r_data, lead_field, neigh, neigh_p,
               s_noise, sigma_q, lam, N_dip_max):
        """Perform a full evolution step of the whole empirical pdf.
        This is done by calling the evol_n_dips method on each particle
        forming the empirical pdf and calling the evol_loc method on
        each dipole of each particle forming the empirical pdf.

        Parameters
        ----------
        n_verts : int
            The number of the points in the given brain discretization.
        r_data : array of floats, shape (n_sens, n_ist)
            The real part of the data; n_sens is the number of sensors and
            n_ist is the number of time-points.
        lead_field : array of floats, shape (n_sens x 3*n_verts)
            The leadfield matrix.
        neigh : array of ints
            The neighbours of each point in the brain discretization.
        neigh_p : array of floats
            The neighbours' probabilities.
        s_noise : float
            The standard deviation of the noise distribution.
        sigma_q : float
            The standard deviation of the prior of the dipole moment
        lam : float
            The parameter of the prior Poisson pdf of the number of dipoles.
        N_dip_max : int
            The maximum number of dipoles allowed in each particle forming the
            empirical pdf.

        """

        for i_part, _part in enumerate(self.particles):
            _part = _part.evol_n_dips(n_verts, r_data, lead_field, N_dip_max,
                                      self.exponents[-1], s_noise, sigma_q,
                                      lam)
            for dip_idx in reversed(range(_part.n_dips)):
                _part = _part.evol_loc(dip_idx, neigh, neigh_p, r_data,
                                       lead_field, self.exponents[-1], s_noise,
                                       sigma_q, lam)
            self.particles[i_part] = _part
            # TODO: more clever way?

    def resample(self):
        """Performs a systematic resampling step of the whole empirical pdf
         in which the particles having small normalized importance weights
         are most likely discarded whereas the best particles are replicated
         in proportion to their importance weights. This is done in order to
         prevent the degeneracy of the sample (namely the circumstance in which
         all but one particle have negligible weights).
        """
        weights = np.exp(self.logweights)
        w_part = np.cumsum(weights)

        # ------------------------------------
        w_part[-1] = 1
        w_part[np.where(w_part > 1)] = 1
        # ------------------------------------

        u_part = (np.arange(weights.shape[0], dtype=float) +
                  np.random.uniform()) / weights.shape[0]

        new_ind = np.digitize(u_part, w_part)
        new_ind_ord = np.array(sorted(list(new_ind),
                               key=list(new_ind).count, reverse=True))
        self.particles = self.particles[new_ind_ord]
        self.logweights[:] = np.log(1. / self.logweights.shape[0])
        self.ESS = self.logweights.shape[0]

    def compute_exponent(self, s_noise, gamma_high=0.99, gamma_low=0.9,
                         delta_min=1e-05, delta_max=0.1):
        """The choice for the sequence of artificial distributions  consists
        in starting from the prior distribution and moving towards the
        posterior by increasing the exponent of the likelihood function with
        the iterations.

        This method computes the exponent to be used in the next iteration in
        an "adaptive" manner in order to optimize the trade-off between the
        computational speed and the quality of the approximation.
        Moreover, the method updates the particle weights.

        Parameters
        ----------
        s_noise : float
            The standard deviation of the noise distribution.
        """
        if self.exponents[-1] == 1:
            if self.verbose:
                print('Last iteration...')
            self.exponents = np.append(self.exponents, 1.01)
        else:
            delta_a = delta_min
            delta_b = delta_max
            delta = delta_max
            ESS_new = 0
            iterations = 1
            last_op_incr = False

            while ESS_new/self.ESS > gamma_high or \
                    ESS_new/self.ESS < gamma_low:

                # log of the unnormalized weights
                log_weights_aux = np.array([self.logweights[i_part] + (delta/(2*s_noise**2)) *
                                            _part.loglikelihood_unit for i_part, _part in enumerate(self.particles)])
                # normalization
                w = log_weights_aux.max()
                log_weights_aux = log_weights_aux - w - \
                    np.log(np.exp(log_weights_aux - w).sum())
                # Actual weights:
                weights_aux = np.exp(log_weights_aux)

                ESS_new = np.float32(1. / np.square(weights_aux).sum())

                if ESS_new / self.ESS > gamma_high:
                    delta_a = delta
                    delta = min([(delta_a + delta_b)/2, delta_max])
                    last_op_incr = True
                    if (delta_max - delta) < delta_max/100:
                        # log of the unnormalized weights
                        log_weights_aux = np.array([self.logweights[i_part] + (delta/(2*s_noise**2)) *
                                                    _part.loglikelihood_unit
                                                    for i_part, _part in enumerate(self.particles)])
                        # normalization
                        w = log_weights_aux.max()
                        log_weights_aux = log_weights_aux - w - \
                            np.log(np.exp(log_weights_aux - w).sum())
                        # Actual weights:
                        weights_aux = np.exp(log_weights_aux)
                        break
                elif ESS_new / self.ESS < gamma_low:
                    delta_b = delta
                    delta = max([(delta_a + delta_b)/2, delta_min])
                    if (delta - delta_min) < delta_min/10 or \
                            (iterations > 1 and last_op_incr):
                        # log of the unnormalized weights
                        log_weights_aux = np.array([self.logweights[i_part] + (delta/(2*s_noise**2)) *
                                                    _part.loglikelihood_unit
                                                    for i_part, _part in enumerate(self.particles)])
                        # normalization
                        w = log_weights_aux.max()
                        log_weights_aux = log_weights_aux - w - \
                            np.log(np.exp(log_weights_aux - w).sum())
                        # Actual weights:
                        weights_aux = np.exp(log_weights_aux)
                        break
                    last_op_incr = False

                iterations += 1

            if self.exponents[-1] + delta > 1:
                delta = 1 - self.exponents[-1]
                # log of the unnormalized weights
                log_weights_aux = np.array([self.logweights[i_part] + (delta/(2*s_noise**2)) *
                                            _part.loglikelihood_unit for i_part, _part in enumerate(self.particles)])
                # normalization
                w = log_weights_aux.max()
                log_weights_aux = log_weights_aux - w - \
                    np.log(np.exp(log_weights_aux - w).sum())
                # Actual weights:
                weights_aux = np.exp(log_weights_aux)

            self.exponents = np.append(self.exponents, self.exponents[-1] + delta)
            self.logweights = log_weights_aux
            self.ESS = np.float32(1. / np.square(weights_aux).sum())

    def point_estimate(self, D, N_dip_max):
        """Computation of a point estimate for the number of dipoles and their
        parameters from the empirical distribution

        Parameters
        ----------
        V : float 2D array of shape n_vertices x 3
            Source space
        N_dip_max : int
            maximum allowed number of dipoles in the Particle instance

        Returns
        -------
        self : EmpPdf istance
            updated EmpPdf
        """

        if self.verbose:
            print('Computing estimates...')
        weights = np.exp(self.logweights)

        # Step1: Number of Dipoles
        #    1a) Compute model_selection
        self.model_sel = np.zeros(N_dip_max+1)

        for i_p, _part in enumerate(self.particles):
            if _part.n_dips <= N_dip_max:
                self.model_sel[_part.n_dips] += weights[i_p]

        #     1b) Compute point estimation
        self.est_n_dips = np.argmax(self.model_sel)

        # Step2: Positions of the dipoles
        if self.est_n_dips == 0:
            self.est_locs = np.array([])
            self.blob = np.array([])
        else:
            nod = np.array([_part.n_dips for _part in self.particles])
            selected_particles = np.delete(self.particles, np.where(nod != self.est_n_dips))
            selected_weights = np.delete(weights, np.where(nod != self.est_n_dips))
            ind_bestpart = np.argmax(selected_weights)
            bestpart_locs = np.array([_dip.loc for _dip in selected_particles[ind_bestpart].dipoles])
            order_dip = np.empty([selected_particles.shape[0], self.est_n_dips], dtype='int')

            all_perms_index = np.asarray(list(itertools.permutations(range(self.est_n_dips))))

            for i_p, _part in enumerate(selected_particles):
                part_locs = np.array([_dip.loc for _dip in _part.dipoles])
                OSPA = np.mean(D[part_locs[all_perms_index], bestpart_locs], axis=1)
                bestperm = np.argmin(OSPA)
                order_dip[i_p] = all_perms_index[bestperm]

            self.blob = np.zeros([self.est_n_dips, D.shape[0]])

            for dip_idx in range(self.est_n_dips):
                for i_p, _part in enumerate(selected_particles):
                    loc = _part.dipoles[order_dip[i_p, dip_idx]].loc
                    self.blob[dip_idx, loc] += selected_weights[i_p]

            self.est_locs = np.argmax(self.blob, axis=1)


class Sesame(object):
    """ Sequential Semi-Analytic Monte-Carlo Estimation (SESAME) of sources

    Parameters
    ----------
    forward : dict
        Forward operator
    evoked : instance of Evoked
        The evoked data
    s_noise : float
        The standard deviation of the noise distribution.
    radius : float | None (default None)
        The maximum distance (in cm) that is allowed between two point of
        the source space to be considered neighbours.
        If None it is set equal to 1 cm.
    sigma_neigh: float | None (default None)
        Standard deviation of the probability distribution of neighbours.
        If None it is set equal to radius/2.
    n_parts : int (default 100)
        The number of particles forming the empirical pdf.
    sample_min : float | None (default None)
        First sample of the time window in which data are analyzed.
        If None, time window starts from the first sample of the data.
    sample_max : float | None (default None)
        Last sample of the time window in which dara are analyzed.
        If None, time window ends with the last sample of the data.
    s_q : float | None (default None)
        The standard deviation of the prior of the dipole moment.
        If None its value is automatic estimated.
    lam : float (default 0.25)
        The parameter of the prior Poisson pdf of the number of dipoles.
    N_dip_max : int (default 10)
        The maximum number of dipoles allowed in each particle.

    Attributes
    ----------
    lam : float
        The parameter of the prior Poisson pdf of the number of dipoles.
    lead_field : array of floats, shape (n_sens x 3*n_verts)
        The leadfield matrix.
    source_space : array of floats, shape  (n_verts, 3)
        The coordinates of the points in the brain discretization.
    forward : dict
        The forward structure.
    neigh : array of ints
        The neighbours of each point in the brain discretization.
    neigh_p : array of floats
        The neighbours' probabilities.
    s_min : int
        The first sample of the time window in which data are analyzed.
    s_max : int
        The last sample of the time window in which data are analyzed.
    r_data : array of floats, shape (n_sens, n_ist)
        The real part of the data; n_sens is the number of sensors and
        n_ist is the number of time-points or of frequencies.
    i_data : array of floats, shape (n_sens, n_ist)
        The imaginary part of the data; n_sens is the number of sensors
        and n_ist is the number of time-points or of frequencies.
    emp : instance of EmpPdf
        The empirical pdf approximated by the particles at each iteration.
    _resample_it : list of ints
        The iterations during which a resampling step has been performed
    ESS : list of floats
        The Effective Sample Size over the iterations.
    model_sel : list of arrays of floats
        The model selection (i.e. the posterior distribution of the number
        of dipoles) over the iterations.
    est_n_dips : list of ints
        The estimated number of dipoles over the iterations.
    blob : list of 2D arrays of floats
        The intensity measure of the point process over the iterations.
    est_locs : list of array of ints
        The estimated source locations over the iterations.
    est_q : array of floats, shape (n_ist x (3*est_n_dips[-1]))
        The sources' moments estimated at the last iteration.
    gof : float
        The goodness of fit at a fixed iteration, i.e.
                gof = 1 - ||meas_field - rec_field|| / ||meas_field||
        where:
        meas_field is the recorded data,
        rec_field is the reconstructed data,
        and || || is the Frobenius norm.
    """

    def __init__(self, forward, evoked, s_noise, radius=None, sigma_neigh=None,
                 n_parts=100, sample_min=None, sample_max=None, subsample=None,
                 s_q=None, cov=None, lam=0.25, N_dip_max=10, verbose=False):

        # 1) Choosen by the user
        self.n_parts = n_parts
        self.lam = lam
        self.N_dip_max = N_dip_max
        self.verbose = verbose
        self.forward, _info_picked = _select_orient_forward(forward, evoked.info, cov)

        self.source_space = forward['source_rr']
        self.n_verts = self.source_space.shape[0]
        self.lead_field = forward['sol']['data']

        self.distance_matrix = ssd.cdist(self.source_space, self.source_space)
        if radius is None:
            self.radius = self.initialize_radius()
        else:
            self.radius = radius
        print('Computing neighbours matrix...')
        self.neigh = self.create_neigh(self.radius)
        print('[done]')

        if sigma_neigh is None:
            self.sigma_neigh = self.radius/2
        else:
            self.sigma_neigh = sigma_neigh
        print('Computing neighbours probabilities...')
        self.neigh_p = self.create_neigh_p(self.sigma_neigh)
        print('[done]')

        if sample_min is None:
            self.s_min = 0
        else:
            if isinstance(sample_min, (int, np.integer)):
                self.s_min = sample_min
            else:
                raise ValueError('sample_min index should be an integer')
            # self.ist_in = np.argmin(np.abs(evoked.times-time_in * 0.001))
            # TODO: pensare meglio alla definizione di distanza (istante piu' vicino? o istante prima/dopo?)
        if sample_max is None:
            self.s_max = evoked.data.shape[1]-1
        else:
            # self.ist_fin = np.argmin(np.abs(evoked.times - time_fin * 0.001))
            if isinstance(sample_max, (int, np.integer)):
                self.s_max = sample_max
            else:
                raise ValueError('sample_max index should be an integer')
        print('Analyzing data from {0} ms to {1} ms'.format(round(evoked.times[self.s_min], 4),
                                                            round(evoked.times[self.s_max], 4)))

        if subsample is not None:
            self.subsample = subsample

        if subsample is not None:
            print('Subsampling data with step {0}'.format(subsample))
            data = evoked.data[:, self.s_min:self.s_max + 1:subsample]
        else:
            data = evoked.data[:, self.s_min:self.s_max+1]

        # Perform whitening if a noise covariance is provided
        if cov is not None:
            whitener, _ = compute_whitener(cov, info=_info_picked, pca=True,
                                           picks=_info_picked['ch_names'])
            data = np.sqrt(evoked.nave) * np.dot(whitener, data)
            self.lead_field = np.sqrt(evoked.nave) * np.dot(whitener, self.lead_field)

        self.r_data = data.real
        self.i_data = data.imag
        del data

        if s_q is None:
            print('Estimating dipole strength variance...')
            self.s_q = self.estimate_s_q()
            print('[done]')
            print(' Estimated dipole strength variance: {:.4e}'.format(self.s_q))
        else:
            self.s_q = s_q

        if s_noise is None:
            print('Estimating noise variance...')
            self.s_noise = self.estimate_s_noise()
            print('[done]')
            print(' Estimated noise variance: {:.4e}'.format(self.s_noise))
        else:
            self.s_noise = s_noise

        self._resample_it = list()
        self.est_n_dips = list()
        self.est_locs = list()
        self.est_q = np.array([])
        self.model_sel = list()
        self.blob = list()
        self.SW_pv = list()

        self.emp = EmpPdf(self.n_parts, self.n_verts,
                          self.lam, verbose=self.verbose)

        for _part in self.emp.particles:
            _part.compute_loglikelihood_unit(self.r_data, self.lead_field,
                                             self.s_noise, self.s_q)

    def apply_sesame(self, estimate_q=True):
        """Run the SASMC sampler algorithm and performs point estimation at
         the end of the main loop.

        Parameters
        ----------
        estimate_q : bool
            If true a point-estimate of the dipole moment is computed at the
            last iteration
        """

        print('Computing inverse solution. This will take a while...')
        # --------- INIZIALIZATION ------------
        # Samples are drawn from the prior distribution and weigths are set as
        # uniform.
        nd = np.array([_part.n_dips for _part in self.emp.particles])

        # Creation of distances matrix
        # D = ssd.cdist(self.source_space, self.source_space)

        while not np.all(nd <= self.N_dip_max):
            nd_wrong = np.where(nd > self.N_dip_max)[0]
            self.emp.particles[nd_wrong] =\
                np.array([Particle(self.n_verts, self.lam)
                         for _ in itertools.repeat(None, nd_wrong.shape[0])])
            nd = np.array([_part.n_dips for _part in self.emp.particles])

        # Point estimation for the first iteraction
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
                            self.s_q, self.lam, self.N_dip_max)

            # STEP 3: Point Estimation
            # self.emp.point_estimate(D, self.N_dip_max)
            #
            # self.est_n_dips.append(self.emp.est_n_dips)
            # self.model_sel.append(self.emp.model_sel)
            # self.est_locs.append(self.emp.est_locs)
            # self.blob.append(self.emp.blob)

            # STEP 4: compute new exponent e new weights
            self.emp.compute_exponent(self.s_noise)

            time.sleep(0.01)
            time_elapsed = (time.time() - time_start)
            if self.verbose:
                print('Computation time: {:.2f} seconds'.format(time_elapsed))
                print('-------------------------------')

        # Estimation
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

    def initialize_radius(self):
        x_length = np.amax(self.source_space[:, 0]) - np.amin(self.source_space[:, 0])
        y_length = np.amax(self.source_space[:, 1]) - np.amin(self.source_space[:, 1])
        z_length = np.amax(self.source_space[:, 2]) - np.amin(self.source_space[:, 2])

        max_length = max(x_length, y_length, z_length)

        if max_length > 50:
            radius = 10
        elif max_length > 1:
            radius = 1
        else:
            radius = 0.01

        return radius

    def create_neigh(self, radius):
        n_max = 100
        n_min = 3
        # D = ssd.cdist(self.source_space, self.source_space)
        reached_points = np.array([0])
        counter = 0
        n_neigh = []
        list_neigh = []

        while counter < reached_points.shape[0] and self.source_space.shape[0] > reached_points.shape[0]:
            P = reached_points[counter]
            aux = np.array(sorted(np.where(self.distance_matrix[P] <= radius)[0],
                                  key=lambda k: self.distance_matrix[P, k]))
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
        elif self.source_space.shape[0] == reached_points.shape[0]:
            while counter < self.source_space.shape[0]:
                P = reached_points[counter]
                aux = np.array(sorted(np.where(self.distance_matrix[P] <= radius)[0],
                                      key=lambda k: self.distance_matrix[P, k]))
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

            # n_neigh_min = min(n_neigh)
            # n_neigh_mean = sum(n_neigh) / len(n_neigh)
            # print('***** Tested radius = ' + str(radius) + ' *****')
            # print('Maximum number of neighbours: ' + str(n_neigh_max))
            # print('Minimum number of neighbours: ' + str(n_neigh_min))
            # print('Average number of neighbours: ' + str(n_neigh_mean))

            neigh = np.zeros([self.source_space.shape[0],
                              n_neigh_max], dtype=int) - 1
            for i in range(self.source_space.shape[0]):
                neigh[i, 0:list_neigh[i].shape[0]] = list_neigh[i]
            index_ord = np.argsort(neigh[:, 0])
            neigh = neigh[index_ord]
            return neigh

        else:
            raise RuntimeError('Some problems during computation of neighbours.')

    def create_neigh_p(self, sigma_neigh):
        # D = ssd.cdist(self.source_space, self.source_space)
        neigh_p = np.zeros(self.neigh.shape, dtype=float)
        for i in range(self.source_space.shape[0]):
            n_neig = len(np.where(self.neigh[i] > -1)[0])
            neigh_p[i, 0:n_neig] = \
                np.exp(-self.distance_matrix[i, self.neigh[i, 0:n_neig]] ** 2 / (2 * sigma_neigh ** 2))
            neigh_p[i] = neigh_p[i] / np.sum(neigh_p[i])
        return neigh_p

    def estimate_s_q(self):

        s_q = 15 * np.max(abs(self.r_data)) / np.max(abs(self.lead_field))

        return s_q

    def estimate_s_noise(self):

        s_noise = 0.2 * np.max(abs(self.r_data))

        return s_noise

    def compute_q(self, est_locs):
        """Point-estimation of the dipole moment

        Parameters
        ----------
        est_locs : int array
            Estimated dipole location (index of the brain grid points)
        """
        est_num = est_locs.shape[0]
        [n_sens, n_time] = np.shape(self.r_data)

        ind = np.ravel([[3*est_locs[idip], 3*est_locs[idip]+1,
                       3*est_locs[idip]+2] for idip in range(est_num)])
        Gc = self.lead_field[:, ind]
        sigma = (self.s_q / self.s_noise)**2 * np.dot(Gc, np.transpose(Gc)) +\
            np.eye(n_sens)
        kal_mat = (self.s_q / self.s_noise)**2 * np.dot(np.transpose(Gc),
                                                        np.linalg.inv(sigma))
        self.est_q = np.array([np.dot(kal_mat, self.r_data[:, t])
                              for t in range(n_time)])

    def goodness_of_fit(self):
        """Evaluation of the perfomance

        Parameters
        ----------
        iteration : int
            Iteration of interest
        """

        if not hasattr(self, 'est_n_dips'):
            raise AttributeError('None estimation found!!!')

        est_n_dips = self.est_n_dips[-1]
        est_locs = self.est_locs[-1]
        est_q = self.est_q
        meas_field = self.r_data
        rec_field = np.zeros(meas_field.shape)
        for i_d in range(est_n_dips):
            rec_field += np.dot(self.lead_field[:, 3*est_locs[i_d]:3*(est_locs[i_d]+1)],
                                est_q[:, 3*i_d:3*(i_d+1)].T)

        GOF = 1 - np.linalg.norm(meas_field - rec_field) / np.linalg.norm(meas_field)

        return GOF

    def to_stc(self, subject=None):
        """Export results in .stc file

        Parameters
        ----------
        file_name : str
            Path and name of the file to be saved
        fwd : dict
            Forward structure from which the lead-field matrix and the source
            space were been extracted
        it_in and it_fin : int
            First and last iteration to be saved
        subject : str
            Name of the subject
        """
        if 'SourceEstimate' not in dir():
            from mne import SourceEstimate

        if not hasattr(self, 'blob'):
            raise AttributeError('Run filter first!!')

        # TODO: fwd is already saved in _sasmc
        blobs = self.blob
        fwd = self.forward
        vertno = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
        nv_tot = fwd['nsource']

        blob_tot = np.array([np.sum(bl, axis=0) for bl in blobs[1:]])
        # I don't store first iteration which is always empty

        tmin = 1
        stc = SourceEstimate(data=blob_tot.T, vertices=vertno, tmin=tmin,
                             tstep=1, subject=subject)
        return stc

    @mlab.show
    def plot_itensity_measure(self, subject, subjects_dir, estimated_loc=True,
                              surface='inflated', hemi='split', views='lat',
                              clim='auto'):
        # TODO: Does it work also when we have multiple iterations?
        # TODO some default value for subject/subject_dir?
        # TODO some text in time-label?
        # TODO: input as mne-python plot function?
        # TODO: what happen if no dipole are estimate?
        # TODO: --> do we really need a function?
        # TODO: --> when hemi = 'split' problem in keeping the figure.

        stc = self.to_stc(subject)
        brain = stc.plot(subject, surface=surface, hemi=hemi,
                         time_label=' ', subjects_dir=subjects_dir,
                         views=views, clim=clim)
        if estimated_loc:
            est_locs = self.est_locs[-1]
            nv_lh = stc.vertices[0].shape[0]
            est_locs_lh = est_locs[np.where(est_locs < nv_lh)[0]]
            est_locs_rh = est_locs[np.where(est_locs >= nv_lh)[0]] - nv_lh
            if (hemi is not 'rh') and (est_locs_lh.size != 0):
                brain.add_foci(stc.vertices[0][est_locs_lh],
                               coords_as_verts=True, hemi='lh',
                               color='blue', scale_factor=0.3)
            if (hemi is not 'lh') and (est_locs_rh.size != 0):
                brain.add_foci(stc.vertices[1][est_locs_rh],
                               coords_as_verts=True, hemi='rh',
                               color='blue', scale_factor=0.3)
        return brain

    def write_stc(self, file_name, fwd, tmin=1, tmax=None, subject=None):

        # TODO: --> Do we really need this function?

            """Export results in .stc file

            Parameters
            ----------
            file_name : str
                Path and name of the file to be saved
            fwd : dict
                Forward structure from which the lead-field
                matrix and the source space have been extracted
            it_in and it_fin : int
                First and last iteration to be saved
            subject : str
                Name of the subject
            """

            stc = self.to_stc(fwd, tmin, tmax, subject)
            stc.save(file_name)
