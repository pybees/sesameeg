.. |br| raw:: html

    <br>

##################
Mathematical model
##################

SESAME (Sequential Semi-Analytic Montecarlo Estimation) employs a Bayesian perspective on the problem of
estimating an unknown number of current dipoles from a set of spatial topographies of
MagnetoEncephaloGraphic (MEG) and/or ElectroEncephaloGraphic (EEG) data.
This section intends to outline the main ideas behind it.
For a thorough description of the subject, the reader is referred to [1]_, [2]_.


For the sake of clarity, the description below deals with the analysis of a single MEG/EEG topography.
However, as shown in [1]_, this approach easily generalizes to include multiple topographies
under the hypothesis that both the number of sources and their locations do not change.

Multi--dipole source model.
---------------------------
SESAME makes use of the equivalent current dipole model for the neuronal currents.
In this framework, the brain volume is discretized into small domains and the activity of the neuronal population
inside any of these domains is represented by a point source, which can be thought of as the concentration of
the current to a given reference point of the domain.

In mathematical terms, each of these point sources, termed *current dipoles*, is an applied vector, whose
moment :math:`\mathbf{q}` expresses the strength and the orientation of the current.
The neuronal primary current distribution :math:`\mathbf{j}` is then assumed to be  closely approximated by
the superposition of a small --- but unknown --- number :math:`n_D`
of current dipoles, and can therefore be seen in abstract terms as a point in the disjoint union of
spaces

.. math:: \mathcal{J}\ :=\ \bigcup_{n_D}\ \{n_D\} \times \mathcal{D}^{n_D}

in which :math:`\mathcal{D}^{n_D}` is the state space of the  :math:`n_D`--tuple of current dipoles
approximating :math:`\mathbf{j}`, and where the number :math:`n_D` of dipoles is explicitly included
among the unkowns. The space :math:`\mathcal{D}^{n_D}` is defined as the Cartesian product of
:math:`n_D` copies of the single dipole space :math:`\mathcal{D}` whose points
are given by the pairs  :math:`(r, \mathbf{q})`, in which :math:`r` is an integer variable
representing the dipole location and :math:`\mathbf{q}` is a three--dimensional vector representing
the dipole moment.
Any current distribution :math:`\mathbf{j}` is therefore represented as

.. math:: \mathbf{j} = \left(n_D, r_1, \mathbf{q}_1, \dots, r_{n_D}, \mathbf{q}_{n_D} \right) \in \mathcal{J}\, .

or also equivalently as

.. math:: \mathbf{j} = \left(\/n_D, \mathbf{r}_{1:n_D}, \mathbf{q}_{1:n_D}\/\right)\ ,

which directly follows from the previous equation by reordering the axes and by introducing the shorthand notations

.. math:: \mathbf{r}_{1:n_D}\, :=\, \left(r_1, \ldots, r_{n_D}\right),
.. math:: \mathbf{q}_{1:n_D}\, :=\, \left(\mathbf{q}_{\/1}, \ldots, \mathbf{q}_{\/{n_D}}\right).

Measurement model
-----------------
Let :math:`\mathbf{y}_t = (\/{y_t}^{\!1}, \ldots, {y_t}^{\!n_s}\/)` denote the data recorded by
the :math:`n_S` MEG/EEG sensors at time :math:`\ t\ `.
Assuming data to be affected by zero--mean Gaussian additive noise, at each sampled time :math:`\ t\ `
the following functional relation holds

.. math:: \mathbf{y}_t = \mathbf{e}_t + \mathbf{n}_t,
  :label: fwd

being :math:`\mathbf{e}_t` the exact field produced by the neural current distribution
:math:`\mathbf{j}_t` and :math:`\mathbf{n}_t` the noise term.
The explicit model for :math:`\mathbf{e}_t` is given by

.. math:: \mathbf{e}_t = \sum_{k=1}^{n_D} G(r_k) \cdot \mathbf{q}_{k,t} =:  G\!\left(\mathbf{r}\/\right) \cdot \mathbf{q}_{t},

where, at time :math:`t\/`,  :math:`G(r_k)` is the lead field matrix computed at the location :math:`r_k` of the
:math:`k`--th dipole on the discretized cortex, :math:`\mathbf{q}_{\/k,t}` is the corresponding dipole moment, and


.. math:: \mathbf{q}_{t}\,  :=\, \left(\mathbf{q}_{\/1,t}, \ldots, \mathbf{q}_{\/{n_D}, t}\right) ,
.. math:: G\!\left(\/\mathbf{r}\/\right)\, :=\, \left[G(r_1)\lvert\,\cdots\lvert G(r_{n_D})\right].

In the forward model both free- (:math:`G(r_k) \in \mathbb{R}^3`) and fixed- (:math:`G(r_k) \in \mathbb{R}^1`, normal
to the cortical surface) dipole orientations are allowed.

Statistical model.
------------------
In a Bayesian approach to the neuromagnetic inverse problem, the MEG/EEG data :math:`\mathbf{y}`,
the unknown :math:`\mathbf{j}` and the noise :math:`\mathbf{e}` are considered as the realizations of
corresponding random variables :math:`\mathbf{Y}`,
:math:`\mathbf{J} = \left(N_D,\, \mathbf{R}_{1:n_D},\, \mathbf{Q}  \,\right)` and :math:`\mathbf{E}`,
related by

.. math:: \mathbf{Y} =  G\!\left(\mathbf{R}\right) \cdot \mathbf{Q} + \mathbf{N} .



In this framework, the solution is the posterior probability density function (pdf) of :math:`\mathbf{J}` conditioned
on the data, which, in the light of Bayes' theorem, can be written as

.. math:: p(\mathbf{j}|\mathbf{y}) \propto p(\mathbf{y}|\mathbf{j})\, p(\mathbf{j})\ ,

being :math:`p(\mathbf{j})` the prior pdf, and :math:`p(\mathbf{y}|\mathbf{j})` the likelihood function.
From :math:`p(\mathbf{j}|\mathbf{y})` sensible estimates of :math:`\mathbf{j}` can then be computed.

Prior distribution.
"""""""""""""""""""
The prior pdf :math:`p(\mathbf{j})` encodes all the information on the unknown which is available before the
measurement is made. Here we set:

.. math:: p(\mathbf{j}) = p(n_D, \mathbf{r}_{1:n_D}, \mathbf{q}_{1:n_D}) = p(n_D) \prod_{k=1}^{n_D}\, p(r_k|n_D, r_1, \ldots, r_{k-1})\, p(\mathbf{q}_{k}),

where:

- :math:`p(n_D)` is the prior pdf for the number of dipole, which is defined as a Poisson distribution with
  mean :math:`\lambda`.
- :math:`p(r_k|n_D, r_1, \ldots, r_{k-1})` is the prior pdf for the location of the :math:`k-` th dipole,
  which is defined as a uniform distribution on the given brain discretization, under the constraint
  that at each grid point can be located at most one dipole.
- :math:`p(\mathbf{q}_k)` is the prior pdf for the dipole moment. Its definition depends on the value given to the boolean
  parameter ``hyper_q`` when instantiating the class :py:class:`~sesameeg.Sesame`.
  In particular:

  * if  ``hyper_q = True``, it is given by a hierarchical model :math:`p(\mathbf{q}_k) = \int p(\mathbf{q}_k|\sigma_q) p(\sigma_q) d\sigma_q`,
    where the conditional distribution :math:`p(\mathbf{q}_k|\sigma_q)` is a trivariate Gaussian distribution with zero
    mean and diagonal matrix equal to :math:`\sigma_q^2 \mathbf{I}` and the prior distribution of the standard deviation
    :math:`\sigma_q` is log-uniform;
  * if  ``hyper_q = False``, it is a trivariate Gaussian distribution with zero mean and diagonal matrix equal
    to :math:`\sigma_q^2 \mathbf{I}`. The variance :math:`\sigma_q^2` reflects information on the dipole strenght.

Likelihood function.
""""""""""""""""""""
The likelihood function, :math:`p(\mathbf{y}|\mathbf{j})`, contains information on the forward model :eq:`fwd` and the
statistical properties of the noise. Here we assume the noise to be Gaussian with zero mean and diagonal covariance
matrix :math:`\sigma_e^2 \mathbf{I}`, thus

.. math:: p(\mathbf{y}|\mathbf{j}) = \mathcal{N}(\mathbf{y}; \mathbf{G} \left( \mathbf{r}_{1:n_D} \right) \cdot \mathbf{q}_{1:n_D}, \sigma_{e}^2 \mathbf{I}).


|

SESAME in action.
-----------------
In order to compute estimates of the unknown neural currents from the posterior distribution, a numerical approximation
of the latter is needed. By exploiting the semi--linear structure of the MEG/EEG forward model SESAME approximates
the posterior pdf

.. math:: p(\mathbf{j}|\mathbf{y})\, =\,  p(\mathbf{q}_{1:n_D}\,|\,\mathbf{y}, n_D, \mathbf{r}_{1:n_D})\ p(n_D, \mathbf{r}_{1:n_D}\,|\,\mathbf{y})

through a two--step approach:

#. first the marginal posterior :math:`p(n_D, \mathbf{r}_{1:n_D}\,|\,\mathbf{y})` is approximated via an
   Adaptive Sequential Monte Carlo (ASMC)  sampler [2]_ ;
#. then  :math:`p(\mathbf{q}_{1:n_D}\,|\,\mathbf{y}, n_D, \mathbf{r}_{1:n_D})` is analytically computed.


ASMC sampler.
"""""""""""""
The ASMC sampler aims at approximating the target pdf :math:`p(n_D, \mathbf{r}_{1:n_D}\,|\,\mathbf{y})` using a large
set of samples, termed `particles`; in our context each particle is a candidate solution and contains all the
parameters that are estimated through the Monte Carlo procedure, namely the number of active sources and their
location.

One easy way to produce such set of samples is to draw them independently from a simple pdf, and possibly weight
them to correctly approximate the target pdf (Importance Sampling, IS [3]_).

An alternative approach is to start from a random candidate, perturb it randomly many times, and then approximate
the target pdf with the collection of samples along the iterations (Markov Chain Monte Carlo, MCMC [3]_).

The ASMC sampler combines these two techniques: a sequence of artificial distributions is defined that smoothly
moves from a tractable prior pdf :math:`p(n_D, \mathbf{r}_{1:n_D})` to the posterior pdf
:math:`p(n_D, \mathbf{r}_{1:n_D}\,|\,\mathbf{y})`, multiple samples are independently drawn from the prior pdf,
evolve following an MCMC scheme, and their weights are updated after every MCMC step;
at times, a resample move is performed, that means samples having negligible weights are replaced by samples in
the higher--probability region, so as to explore better these areas.
Eventually, the target distribution is approximated by the weighted sample set obtained at the last iteration.

The step with which the path from the prior to the posterior pdf is covered is not established a priori,
but adaptively determined at run-time. This means that the actual number of iterations is also determined
online, even if it is always kept within given lower and upper bounds.


Analytic computation of :math:`p(\mathbf{q}_{1:n_D}\,|\,\mathbf{y}, n_D, \mathbf{r}_{1:n_D})`.
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

By exploiting the mutual independence of :math:`\mathbf{J}` and :math:`\mathbf{N}` and the Gaussian assumptions made
about the prior pdf of the dipole moments and the noise model, SESAME analytically compute the posterior
pdf :math:`p(\mathbf{q}_{1:n_D}\,|\,\mathbf{y}, n_D, \mathbf{r}_{1:n_D})`.
Indeed, it is a Gaussian density whose mean and variance depend only on the data, the forward
solution :math:`\mathbf{G}\left(\mathbf{r}_{1:n_D}\right)`, and the standard
deviations :math:`\sigma_q` and :math:`\sigma_e`.

Get the most out of SESAME's results.
-------------------------------------
As descibed above, SESAME approximates the full posterior distribution :math:`p(\mathbf{j}|\mathbf{y})`
as the set of weighted particles

.. math::  \left\{\big(n_D^{i}, \mathbf{r}_{1:n_D^{i}}^{i}\big), w^{i} \right\}_{i=1, ..., I}.


Roughly speaking, each of the :math:`I` particles represents a candidate source configuration,
while the corresponding weight :math:`w^i` quantifies its probability.

SESAME also provides an estimate of the unknown neuronal primary current distribution :math:`\mathbf{j}`,
through the following procedure:

* first the most probable model is identified by estimating mode of the posterior pdf for the number of sources i.e.

  .. math:: \hat{n}_D = argmax\, p \left(n_D | \mathbf{y} \right) = argmax\, \sum_{i=1}^I w^{i} \delta \left(n_D-n_D^{i} \right);

* subsequently, for each point :math:`r` in the cortical discretization, the posterior probability of a source
  being located in :math:`r` is computed as:

  .. math:: p(r| \mathbf{y},\hat{n}_D) = \sum_{i=1}^I w^i \delta\left(\hat{n}_D,n_D^i\right) \sum_{k=1}^{n_D^{i}} \delta\left(r, r_k^{i}\right)\, ;

* the above quantity is then used to produce posterior maps of activation on the cortical surface and to compute
  estimates of dipole locations as the local peaks of such a probability map;

* finally, dipole moments can be reasonably estimated as the mean of the corresponding Gaussian distribution.

.. note::
    The number of components of the estimated dipole moments depends on the dipole orientation constraint in the forward
    model.

References
----------
.. [1] S. Sommariva and A. Sorrentino, `Sequential Monte Carlo samplers for semi-linear inverse problems and application to Magnetoencephalography <https://doi.org/10.1088/0266-5611/30/11/114020>`_. Inverse Problems, 30 114020 (2014).
.. [2] A. Sorrentino, G. Luria, and R. Aramini, `Bayesian multi-dipole modeling of a single topography in MEG by adaptive Sequential Monte Carlo Samplers <https://iopscience.iop.org/article/10.1088/0266-5611/30/4/045010>`_. Inverse Problems, 30 045010 (2014).
.. [3] C. Robert and G. Casella, `Monte Carlo Statistical Methods <https://www.springer.com/gp/book/9780387212395>`_, 2nd Edition.  Springer (2004).
