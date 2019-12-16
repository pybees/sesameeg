# -*- coding: utf-8 -*-

# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#          Alberto Sorrentino <sorrentino@dima.unige.it>
#
# License: BSD (3-clause)

import copy
import itertools
import numpy as np
import scipy.special as sps
from .dipoles import Dipole
from .utils import woodbury


def gamma_pdf(x, shape, scale):
    """
    Evaluates the Gamma pdf on a single point.
    :param x: float
        The point where to evaluate the Gamma pdf
    :param shape: int
        The Gamma pdf shape parameter :math:`k`
    :param scale: float
        The Gamma pdf scale parameter :math:`\\theta`
    :return:
        The Gamma pdf evaluated on x
    """
    return x**(shape-1)*(np.exp(-x/scale) / (sps.gamma(shape)*scale**shape))


class Particle(object):
    """Particle class for SESAME, used to store a single particle
    of an empirical pdf.

    Parameters
    ----------
    n_verts : int
        The number of the points in the given brain discretization.
    lam : float
        The parameter of the prior Poisson pdf of the number of dipoles.
    s_q : float
        The standard deviation of the prior of the dipole moment.
    hyper_q : bool
        If True use hyperprior in dipole strength

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

    def __init__(self, n_verts, lam, s_q=None, hyper_q=False, q_in=None):
        """Initialization: the initial number of dipoles is Poisson
           distribuited; the initial locations are uniformly distribuited
           within the brain grid points, with no dipoles in the same position.
        """
        self.hyper_q = hyper_q
        self.n_dips = 0
        self.dipoles = np.array([])
        self.prior = None
        self.loglikelihood_unit = None

        if isinstance(s_q, float):
            if self.hyper_q:
                self.s_q = 10 ** (3 * np.random.rand()) * (s_q / 35)
            else:
                self.s_q = s_q

        if isinstance(q_in, float):
            self.q_in = q_in

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

    def _check_sigma(self, r_data, lead_field, s_noise):
        [n_sens, n_ist] = r_data.shape

        # Step 1: compute variance of the likelihood.
        if self.n_dips == 0:
            sigma = np.eye(n_sens)
        else:
            # 1a: compute the leadfield of the particle
            idx = np.ravel([[3 * dip.loc, 3 * dip.loc + 1, 3 * dip.loc + 2]
                            for dip in self.dipoles])
            Gc = lead_field[:, idx]
            # 1b: compute the variance
            sigma = (self.s_q / s_noise) ** 2 * np.dot(Gc, np.transpose(Gc)) + \
                    np.eye(n_sens)

        if np.all(np.linalg.eigvals(sigma) > 0):
            return True
        else:
            return False

    def compute_loglikelihood_unit(self, r_data, lead_field, s_noise=None):
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

        Returns
        -------
        loglikelihood_unit : float
            The logarithm of the likelihood function in the present particle.
        """

        [n_sens, n_ist] = r_data.shape

        # Step 1: compute the covariance of the likelihood.
        if self.n_dips == 0:
            sigma = np.eye(n_sens)
            det_sigma = np.linalg.det(sigma)
            inv_sigma = np.eye(n_sens)
        else:
            # 1a: compute the leadfield of the particle
            idx = np.ravel([[3*dip.loc, 3*dip.loc+1, 3*dip.loc+2]
                           for dip in self.dipoles])
            Gc = lead_field[:, idx]
            # 1b: compute the covariance
            sigma = (self.s_q / s_noise)**2 * np.dot(Gc, np.transpose(Gc)) + \
                np.eye(n_sens)
            det_sigma = np.linalg.det(sigma)
            inv_sigma = woodbury(np.eye(n_sens), (self.s_q / s_noise)**2 * Gc,
                                 Gc.T, Gc.shape[1])

        # Step 2: compute the log_likelihood
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
        if hasattr(self, 'q_in'):
            self.prior *= np.prod(np.array([1 / np.fabs(dip.re_q)
                                            for dip in self.dipoles])) * \
                          np.prod(np.array([1 / np.fabs(dip.im_q)
                                            for dip in self.dipoles]))
        elif hasattr(self, 's_q') and self.hyper_q is True:
            self.prior /= self.s_q
        return self.prior

    def evol_n_dips(self, n_verts, r_data, lead_field, N_dip_max,
                    lklh_exp, s_noise, lam, q_birth=1/3, q_death=1/20):
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
        lam : float
            Parameter of the Poisson probability distribution used for
            determining the number of dipoles in the particle.
        q_birth : float
            Probability of proposing to add a dipole. We recommend to use
            the default value q_birth = 1/3.
        q_death : float
            Probability of proposing to remove a dipole. We recommend
            to use the default value q_death = 1/20.

        Return
        ------
        self : instance of Particle
            The possibly modified particle instance.
        """

        prop_part = copy.deepcopy(self)
        birth_death = np.random.uniform(1e-16, 1)

        if self.loglikelihood_unit is None:
            self.compute_loglikelihood_unit(r_data, lead_field, s_noise=s_noise)

        if birth_death < q_birth and prop_part.n_dips < N_dip_max:
            prop_part.add_dipole(n_verts)
        elif prop_part.n_dips > 0 and birth_death > 1-q_death:
            sent_to_death = np.random.random_integers(0, self.n_dips-1)
            prop_part.remove_dipole(sent_to_death)

        # Compute alpha rjmcmc
        if prop_part.n_dips != self.n_dips:
            prop_part.compute_prior(lam)
            prop_part.compute_loglikelihood_unit(r_data, lead_field,
                                                 s_noise=s_noise)
            log_prod_like = (prop_part.loglikelihood_unit -
                             self.loglikelihood_unit)

            if prop_part.n_dips > self.n_dips:
                alpha_aux = (q_death * prop_part.prior) / (q_birth * self.prior) * \
                             np.exp((lklh_exp/(2*s_noise**2)) * log_prod_like)

                if hasattr(self, 'q_in'):
                    alpha_aux *= np.fabs(prop_part.dipoles[prop_part.n_dips-1].re_q) * \
                                 np.fabs(prop_part.dipoles[prop_part.n_dips-1].im_q)

                alpha = np.amin([1, alpha_aux])

            elif prop_part.n_dips < self.n_dips:
                alpha_aux = (q_birth * prop_part.prior) / (q_death * self.prior) * \
                            np.exp((lklh_exp/(2*s_noise**2)) * log_prod_like)

                if hasattr(self, 'q_in'):
                    alpha_aux *= np.fabs(self.dipoles[sent_to_death].re_q) * \
                                 np.fabs(self.dipoles[sent_to_death].im_q)

                alpha = np.amin([1, alpha_aux])

            if np.random.rand() < alpha:
                self = copy.deepcopy(prop_part)
        return self

    def evol_loc(self, dip_idx, neigh, neigh_p, r_data, lead_field,
                 lklh_exp, s_noise, lam):
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
        p_part = np.cumsum(
                 neigh_p[prop_part.dipoles[dip_idx].loc,
                         np.where(neigh[prop_part.dipoles[dip_idx].loc] != -1)]
        )
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

        prob_opp_move = neigh_p[
                        prop_loc,
                        np.argwhere(neigh[prop_loc] ==
                                    prop_part.dipoles[dip_idx].loc)[0][0]]
        prop_part.dipoles[dip_idx].loc = prop_loc
        comp_fact_delta_r = prob_opp_move / prob_new_move

        # Compute alpha mcmc
        prop_part.compute_prior(lam)
        prop_part.compute_loglikelihood_unit(r_data, lead_field,
                                             s_noise=s_noise)

        if self.loglikelihood_unit is None:
            self.compute_loglikelihood_unit(r_data, lead_field,
                                            s_noise=s_noise)

        log_prod_like = prop_part.loglikelihood_unit - self.loglikelihood_unit
        alpha = np.amin([1, (comp_fact_delta_r *
                         (prop_part.prior/self.prior) *
                         np.exp((lklh_exp/(2*s_noise**2)) * log_prod_like))])

        if np.random.rand() < alpha:
            self = copy.deepcopy(prop_part)

        return self

    def evol_s_q(self, r_data, lead_field, lklh_exp, s_noise, lam):

        if not hasattr(self, 's_q'):
            raise ValueError

        prop_part = copy.deepcopy(self)

        if self.loglikelihood_unit is None:
            self.compute_loglikelihood_unit(r_data, lead_field, s_noise=s_noise)
        prop_part.s_q = np.random.gamma(3, self.s_q/3)

        # Compute alpha mcmc
        prob_new_move = gamma_pdf(prop_part.s_q, 3, self.s_q/3)
        prob_opp_move = gamma_pdf(self.s_q, 3, prop_part.s_q/3)

        proposal_ratio = prob_opp_move / prob_new_move

        prop_part.compute_prior(lam)
        prop_part.compute_loglikelihood_unit(r_data, lead_field, s_noise=s_noise)

        log_prod_like = prop_part.loglikelihood_unit - self.loglikelihood_unit
        alpha = np.amin([1, (proposal_ratio *
                             (prop_part.prior / self.prior) *
                             np.exp((lklh_exp / (2 * s_noise ** 2)) * log_prod_like))])

        if np.random.rand() < alpha:
            self = copy.deepcopy(prop_part)

        return self
