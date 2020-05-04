# -*- coding: utf-8 -*-

# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#          Alberto Sorrentino <sorrentino@dima.unige.it>
#
# License: BSD (3-clause)

import itertools
import numpy as np
from .particles import Particle


class EmpPdf(object):
    """Empirical probability density function (pdf) class for SESAME.

    Parameters
    ----------
    n_parts : :py:class:`~int`
        The number of particles forming the empirical pdf.
    n_verts : :py:class:`~int`
        The number of the points in the given brain discretization.
    lam : :py:class:`~float`
        The parameter of the prior Poisson pdf of the number of dipoles.
    dip_mom_std : :py:class:`~float`
        The standard deviation of the prior of the dipole moment.
    hyper_q : :py:class:`~bool`
        If True, use hyperprior in dipole strength

    Attributes
    ----------
    particles : :py:class:`~numpy.ndarray` of instances of :py:class:`~particles.Particle`, shape(n_parts,)
        The EmpPdf's particles.
    logweights : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape(n_parts,)
        The logarithm of the weights of the particles forming the
        empirical pdf.
    ESS : :py:class:`~float`
        The Effective Sample Size
    exponents : :py:class:`~numpy.ndarray` of :py:class:`~float`
        Array whose entries represent points in the space of artificial
        distributions. It is used to keep track of the path followed
        by SESAME.
    model_sel : :py:class:`~numpy.ndarray` of :py:class:`~float`
        Marginal posterior probability of the number of sources.
    est_n_dips : :py:class:`~float`
        Estimated number of sources.
    pmap : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape(est_n_dips x n_verts)
        Intensity measure of the point process.
    est_locs : :py:class:`~numpy.ndarray` of :py:class:`~int`
        Estimated sources locations
    """
    def __init__(self, n_parts, n_verts, lam, dip_mom_std=None, hyper_q=False, verbose=False):

        self.hyper_q = hyper_q
        self.particles = np.array([Particle(n_verts, lam, dip_mom_std=dip_mom_std,
                                            hyper_q=self.hyper_q) for _ in itertools.repeat(None, n_parts)])
        self.logweights = np.array([np.log(1/n_parts) for _
                                    in itertools.repeat(None, n_parts)])
        self.ESS = np.float32(1. / np.square(np.exp(self.logweights)).sum())
        self.exponents = np.array([0, 0])
        self.model_sel = None
        self.est_n_dips = None
        self.pmap = None
        self.est_locs = None
        self.verbose = verbose

    def __repr__(self):
        s = ''
        for i_p, _part in enumerate(self.particles):
            s += ("'---- Particle {0} (W = {1},  number of dipoles = {2}):"
                  " \n {3} \n".
                  format(i_p+1, np.exp(self.logweights[i_p]), _part.nu, _part))
        return s

    def sample(self, n_verts, r_data, lead_field, neigh, neigh_p,
               noise_std, lam, max_n_dips):
        """Perform a full evolution step of the whole empirical pdf.

        Parameters
        ----------
        n_verts : :py:class:`~int`
            The number of the points in the given brain discretization.
        r_data : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_sens, n_ist)
            The real part of the data; n_sens is the number of sensors and
            n_ist is the number of time-points.
        lead_field : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_sens x 3*n_verts)
            The leadfield matrix.
        neigh : :py:class:`~numpy.ndarray` of :py:class:`~int`
            The neighbours of each point in the brain discretization.
        neigh_p : :py:class:`~numpy.ndarray` of :py:class:`~float`
            The neighbours' probabilities.
        noise_std : :py:class:`~float`
            The standard deviation of the noise distribution.
        sigma_q : :py:class:`~float`
            The standard deviation of the prior of the dipole moment
        lam : :py:class:`~float`
            The parameter of the prior Poisson pdf of the number of dipoles.
        max_n_dips : :py:class:`~int`
            The maximum number of dipoles allowed in a particle.
        """

        for i_part, _part in enumerate(self.particles):
            if self.hyper_q:
                _part = _part.evol_dip_mom_std(r_data, lead_field, self.exponents[-1], noise_std, lam)

            _part = _part.evol_n_dips(n_verts, r_data, lead_field, max_n_dips,
                                      self.exponents[-1], noise_std, lam)
            for dip_idx in reversed(range(_part.n_dips)):
                _part = _part.evol_loc(dip_idx, neigh, neigh_p, r_data, lead_field,
                                       self.exponents[-1], noise_std, lam)
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

    def compute_exponent(self, noise_std):
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
        noise_std : :py:class:`~float`
            The standard deviation of the noise distribution.
        """

        gamma_high = 0.99
        gamma_low = 0.9
        delta_min = 1e-05
        delta_max = 0.1

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
                log_weights_aux = np.array(
                                  [self.logweights[i_part] +
                                   (delta/(2*noise_std**2)) *
                                   _part.loglikelihood_unit
                                   for i_part, _part
                                   in enumerate(self.particles)])
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
                        log_weights_aux = np.array(
                                          [self.logweights[i_part] +
                                           (delta/(2*noise_std**2)) *
                                           _part.loglikelihood_unit
                                           for i_part, _part
                                           in enumerate(self.particles)])
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
                        log_weights_aux = np.array(
                                          [self.logweights[i_part] +
                                           (delta/(2*noise_std**2)) *
                                           _part.loglikelihood_unit
                                           for i_part, _part
                                           in enumerate(self.particles)])
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
                log_weights_aux = np.array(
                                  [self.logweights[i_part] +
                                   (delta/(2*noise_std**2)) *
                                   _part.loglikelihood_unit
                                   for i_part, _part
                                   in enumerate(self.particles)])
                # normalization
                w = log_weights_aux.max()
                log_weights_aux = log_weights_aux - w - \
                    np.log(np.exp(log_weights_aux - w).sum())
                # Actual weights:
                weights_aux = np.exp(log_weights_aux)

            self.exponents = np.append(self.exponents,
                                       self.exponents[-1] + delta)
            self.logweights = log_weights_aux
            self.ESS = np.float32(1. / np.square(weights_aux).sum())

    def point_estimate(self, D, max_n_dips):
        """Computes a point estimate for the number of active dipoles and
        their locations from the posterior pdf.

        Parameters
        ----------
        D : :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (n_verts x n_verts)
            The Euclidean distance between the points in the
            brain discretization.
        max_n_dips : :py:class:`~int`
            The maximum number of dipoles allowed in a particle.
        """

        if self.verbose:
            print('Computing estimates...')
        weights = np.exp(self.logweights)

        # Step1: Number of Dipoles
        #    1a) Compute model_selection
        self.model_sel = np.zeros(max_n_dips+1)

        for i_p, _part in enumerate(self.particles):
            if _part.n_dips <= max_n_dips:
                self.model_sel[_part.n_dips] += weights[i_p]
            else:
                raise ValueError('Particle {} has too many dipoles!'.format(i_p))

        #     1b) Compute point estimation
        self.est_n_dips = np.argmax(self.model_sel)

        # Step2: Positions of the dipoles
        if self.est_n_dips == 0:
            self.est_locs = np.array([])
            self.pmap = np.array([])
        else:
            nod = np.array([_part.n_dips for _part in self.particles])
            selected_particles = np.delete(self.particles,
                                           np.where(nod != self.est_n_dips))
            selected_weights = np.delete(weights,
                                         np.where(nod != self.est_n_dips))
            ind_bestpar = np.argmax(selected_weights)
            bestpart_locs = np.array([_dip.loc for _dip in
                                      selected_particles[ind_bestpar].dipoles])
            order_dip = np.empty([selected_particles.shape[0],
                                  self.est_n_dips], dtype='int')

            all_perms_index = np.asarray(list(itertools.permutations(
                                              range(self.est_n_dips))))

            for i_p, _part in enumerate(selected_particles):
                part_locs = np.array([_dip.loc for _dip in _part.dipoles])
                OSPA = np.mean(D[part_locs[all_perms_index], bestpart_locs],
                               axis=1)
                bestperm = np.argmin(OSPA)
                order_dip[i_p] = all_perms_index[bestperm]

            self.pmap = np.zeros([self.est_n_dips, D.shape[0]])

            for dip_idx in range(self.est_n_dips):
                for i_p, _part in enumerate(selected_particles):
                    loc = _part.dipoles[order_dip[i_p, dip_idx]].loc
                    self.pmap[dip_idx, loc] += selected_weights[i_p]

            self.est_locs = np.argmax(self.pmap, axis=1)
