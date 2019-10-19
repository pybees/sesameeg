# -*- coding: utf-8 -*-

# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#          Alberto Sorrentino <sorrentino@dima.unige.it>
#
# License: BSD (3-clause)

import numpy as np
import scipy.spatial.distance as ssd
import copy
import time
import itertools
from mne.cov import compute_whitener
from mne.forward import _select_orient_forward
import mne
from mayavi import mlab


class Dipole(object):
    """Single current dipole class for SESAME.

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


class Particle(object):
    """Particle class for SESAME, used to store a single particle 
    of an empirical pdf.

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
        """Add new dipole(s) to the particle.

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
        """Remove a dipole from the particle.

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
        """Evaluates the logarithm of the marginal likelihood in the present particle.

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
            The maximum number of dipoles allowed in a particle.
        lklh_exp : float
            This number represents a point in the sequence of artificial
            distributions used in SESAME.
        s_noise : float
            The standard deviation of the noise distribution.
        s_q : float
            standard deviation of the prior of the dipole moment.
        lam : float
            Parameter of the Poisson probability distribution used for
            determining the number of dipoles in the particle.
        q_birth : float
            Probability of proposing to add a dipole. We recommend to use
            the default value q_birth = 1/3.
        q_death : Probability of proposing to remove a dipole. We recommend
            to use the default value q_death = 1/20.

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
            distributions used in SESAME.
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
    """Empirical probability density function (pdf) class for SESAME.

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
        by SESAME.
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
        For each particle the evol_n_dips method is called and then the
        evol_loc method is applied on each of its dipoles.

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
            The maximum number of dipoles allowed in a particle.
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
        gamma_high, gamma_low : floats
            Upper and lower bound of the acceptable interval for the drop
            in the Effective Sample Size. We recommend to use the default
            values gamma_high=0.99 and gamma_low=0.9.
        delta_min, delta_max : floats
            The minimum and maximum allowed increment of the exponent.
            We recommend to use the default values delta_min=1e-05 and
            delta_max=0.1.
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
        """Computes a point estimate for the number of active dipoles and
        their locations from the posterior pdf.

        Parameters
        ----------
        D : array of floats, shape (n_verts x n_verts)
            The euclidean distance matrix between the points in the
            brain discretization.
        N_dip_max : int
            The maximum number of dipoles allowed in a particle.
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
    """Sequential Semi-Analytic Monte-Carlo Estimation (SESAME) of sources.

    Parameters
    ----------
    forward : instance of Forward
        The forward solution.
    evoked : instance of Evoked
        The evoked data.
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

        if sample_max is None:
            self.s_max = evoked.data.shape[1]-1
        else:
            if isinstance(sample_max, (int, np.integer)):
                self.s_max = sample_max
            else:
                raise ValueError('sample_max index should be an integer')
        print('Analyzing data from {0} ms to {1} ms'.format(round(evoked.times[self.s_min], 4),
                                                            round(evoked.times[self.s_max], 4)))

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
        self.est_q = None
        self.model_sel = list()
        self.blob = list()

        self.emp = EmpPdf(self.n_parts, self.n_verts,
                          self.lam, verbose=self.verbose)

        for _part in self.emp.particles:
            _part.compute_loglikelihood_unit(self.r_data, self.lead_field,
                                             self.s_noise, self.s_q)

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
                np.array([Particle(self.n_verts, self.lam)
                         for _ in itertools.repeat(None, nd_wrong.shape[0])])
            nd = np.array([_part.n_dips for _part in self.emp.particles])

        # Point estimation for the first iteration
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
                            self.s_q, self.lam, self.N_dip_max)

            # STEP 3: Point Estimation
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
        """Guess the units of the points in the brain discretization and
        set to 1 cm the value of the radius for computing the sets of neighbours.

        Returns
        --------
        radius : float
            The value of the radius.
        """

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
        """Compute the set of neighbours for each point of the brain discretization.

        Parameters
        -----------
        radius : float
            The maximum distance between two neighbouring points.

        Returns
        --------
        neigh : array of ints, shape (n_verts, n_neigh_max)
            The sets of neighbours.
        """

        n_max = 100
        n_min = 3
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
        """Compute neighbours' probability.

        Parameters
        -----------
        sigma_neigh : float
            The standard deviation of the Gaussian distribution that defines the
            neighbours' probability.

        Returns
        --------
        neigh_p : array of floats, shape (n_verts, n_neigh_max)
            The neighbours' probability.
        """

        neigh_p = np.zeros(self.neigh.shape, dtype=float)
        for i in range(self.source_space.shape[0]):
            n_neig = len(np.where(self.neigh[i] > -1)[0])
            neigh_p[i, 0:n_neig] = \
                np.exp(-self.distance_matrix[i, self.neigh[i, 0:n_neig]] ** 2 / (2 * sigma_neigh ** 2))
            neigh_p[i] = neigh_p[i] / np.sum(neigh_p[i])
        return neigh_p

    def estimate_s_q(self):
        """Estimate the standard deviation of the prior of the dipole moment.

        Returns
        --------
        s_q : float
            The estimated standard deviation.
        """

        s_q = 15 * np.max(abs(self.r_data)) / np.max(abs(self.lead_field))

        return s_q

    def estimate_s_noise(self):
        """Estimate the standard deviation of noise distribution.

        Returns
        --------
        s_noise : float
            The estimated standard deviation.
        """

        s_noise = 0.2 * np.max(abs(self.r_data))

        return s_noise

    def compute_q(self, est_locs):
        """Compute a point estimate for the dipole moments.

        Parameters
        ----------
        est_locs : list of array of ints
            The estimated source locations.
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
        """Evaluate the estimated configuration of dipoles. The goodness
        of fit (GOF) with the recorded data is defined as

        .. math:: GOF = \\frac{\| \mathbf{y} - \hat{\mathbf{y}} \|}{ \|\mathbf{y}\|}

        where :math:`\mathbf{y}` is the recorded data, :math:`\hat{\mathbf{y}}` is the
        field generated by the estimated configuration of dipoles, and
        :math:`\| \cdot \|` is the Frobenius norm.

        Returns
        -------
        gof : float
            The goodness of fit with the recorded data.
        """

        if len(self.est_n_dips)==0:
            raise AttributeError('No estimation found. Run apply_sesame first.')
        if self.est_q is None:
            raise AttributeError('No dipoles'' moment found. Run compute_q first.')

        est_n_dips = self.est_n_dips[-1]
        est_locs = self.est_locs[-1]
        est_q = self.est_q
        meas_field = self.r_data
        rec_field = np.zeros(meas_field.shape)
        for i_d in range(est_n_dips):
            rec_field += np.dot(self.lead_field[:, 3*est_locs[i_d]:3*(est_locs[i_d]+1)],
                                est_q[:, 3*i_d:3*(i_d+1)].T)

        gof = 1 - np.linalg.norm(meas_field - rec_field) / np.linalg.norm(meas_field)

        return gof

    def to_stc(self, subject=None):
        """Export results in .stc file. Given the estimated number of dipoles :math:`\hat{n}_D`,
        for each point :math:`r` of the brain discretization it computes the posterior
	pdf :math:`p(r|\mathbf{y}, \hat{n}_D)`, i.e. the probability of a source being located
	in :math:`r`.

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
            raise AttributeError('Run filter first!!')

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
