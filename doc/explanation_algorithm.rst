.. |br| raw:: html

    <br>

##################
Mathematical model
##################

SESAME (Sequential Semi-Analytic Montecarlo Estimation) employs a Bayesian perspective on the problem of
estimating an unknown number of current dipoles from a set of spatial topographies of
magnetoencephalografic (MEG) and/or electroencephalografic (EEG) data.
This section intends to outline the main ideas behind the employed source model and algorithm.
For a more precise description we recommend to read our publication [1]_.
| For the sake of clarity, the description below deals with the analysis of a single MEG/EEG topography. However, as shown in [1]_ this approach easily generalizes to include multiple topographies by assuming the number and the locations of the current dipoles to be fixed over topographies, while their strengths and orientations may change.

Multi--dipole source model.
---------------------------
We model the neural sources producing the recorded MEG/EEG data using a primary current distribution which is approximated by the superposition of an unknown number of current dipoles. |br|
In mathematical terms, a source configuration is assumed to belong to the variable--dimension state--space model

.. math:: \mathcal{J}\ :=\ \bigcup_{n_D}\ \{n_D\} \times \mathcal{D}^{n_D} ,
 
were :math:`n_D` is the number of dipoles, :math:`\mathcal{D}` is the state space for a single--dipole configuration and :math:`\mathcal{D}^{n_D}` is the Cartesian product of :math:`n_D` copies of :math:`\mathcal{D}`. |br|
A point in :math:`\mathcal{D}`, i.e. a single--dipole configuration, is a pair :math:`(r, \mathbf{q})` where :math:`r` is an integer variable representing the dipole location in a given discretizion of the cortical surface, and :math:`\mathbf{q}` is a three–dimensional vector representing the dipole moment. Any current distribution :math:`\mathbf{j}` is therefore a set of dipoles represented as 

.. math:: \mathbf{j} = \left(n_D, r_1, \mathbf{q}_1, \dots, r_{n_D}, \mathbf{q}_{n_D} \right) =: \left(n_D, \mathbf{r}_{1:n_D}, \dots, \mathbf{q}_{1:n_D}\right)  \in \mathcal{J}\, .


The functional relation between the current distribution :math:`\mathbf{j}` and the data :math:`\mathbf{y}` recorded by the :math:`n_S` MEG/EEG sensors is given by

.. math:: \mathbf{y} = \mathbf{G} \left( \mathbf{r}_{1:n_D} \right) \cdot \mathbf{q}_{1:n_D} + \mathbf{e},
   :label: fwd

where :math:`\mathbf{e}` is measurement noise, and 

.. math:: \mathbf{G}\left(\mathbf{r}_{1:n_D}\right)\, :=\, \big[\mathbf{G}(r_1)\, \cdots \, \mathbf{G}(r_{n_D})\big]

is a matrix of size :math:`n_S \, \times \, 3N_D` that provides the forward solution associated to the set of dipole locations :math:`r_{1}, \dots, r_{n_D}`. 

Statistical model.
-------------------
| In a Bayesian approach to the dipole estimation problem, the MEG/EEG data :math:`\mathbf{y}`, the unknown :math:`\mathbf{j}` and the noise :math:`\mathbf{e}` are considered as the realizations of corresponding random variables :math:`\mathbf{Y}`, :math:`\mathbf{J}` and :math:`\mathbf{E}`. In this framework, the solution is the posterior probability density function (pdf) of :math:`\mathbf{J}` conditioned on the data, which, in the light of Bayes' theorem, can be written as

.. math:: p(\mathbf{j}|\mathbf{y}) \propto p(\mathbf{y}|\mathbf{j})\, p(\mathbf{j})\ ,

being :math:`p(\mathbf{j})` the prior pdf, and :math:`p(\mathbf{y}|\mathbf{j})` the likelihood function.

Prior distribution.
"""""""""""""""""""
The prior pdf :math:`p(\mathbf{j})` encodes all the information on the unknown source configuration which is available before the measurement is made. Here we set:

.. math:: p(\mathbf{j}) = p(n_D, \mathbf{r}_{1:n_D}, \mathbf{q}_{1:n_D}) = p(n_D) \prod_{k=1}^{n_D}\, p(r_k|n_D, r_1, \ldots, r_{k-1})\, p(\mathbf{q}_{k}),

where:

- :math:`p(n_D)` is the prior pdf for the number of dipole, which is defined as a Poisson distribution with mean :math:`\lambda`.
- :math:`p(r_k|n_D, r_1, \ldots, r_{k-1})` is the prior pdf for the location of the :math:`k-` th dipole, which is defined as a uniform distribution on the given brain discretization excluding the points :math:`r_1, \ldots, r_{k-1}` already occupied by the other dipoles.
- :math:`p(\mathbf{q}_k)` is the prior pdf for the dipole moment, which is a trivariate Gaussian distribution with zero mean and diagonal matrix equal to :math:`\sigma_q^2 \mathbf{I}`. The variance :math:`\sigma_q^2` reflects information on the dipole strenght.

Likelihood function.
"""""""""""""""""""""
The likelihood function, :math:`p(\mathbf{y}|\mathbf{j})`, contains information on the forward model :eq:`fwd` and the statistical properties of the noise. Here we assume the noise to be Gaussian with zero mean and diagonal covariance matrix :math:`\sigma_e^2 \mathbf{I}`, thus

.. math:: p(\mathbf{y}|\mathbf{j}) = \mathcal{N}(\mathbf{y}; \mathbf{G} \left( \mathbf{r}_{1:n_D} \right) \cdot \mathbf{q}_{1:n_D}, \sigma_{e}^2 \mathbf{I}).


Sesame in action.
-----------------
By exploiting the semi--linear structure of the MEG/EEG forward model SESAME approximates the posterior pdf

.. math:: p(\mathbf{j}|\mathbf{y})\, =\,  p(\mathbf{q}_{1:n_D}\,|\,\mathbf{y}, n_D, \mathbf{r}_{1:n_D})\ p(n_D, \mathbf{r}_{1:n_D}\,|\,\mathbf{y})

through a two--step approach: first the marginal posterior :math:`p(n_D, \mathbf{r}_{1:n_D}\,|\,\mathbf{y})` is approximated via an Adaptive Sequential Monte Carlo sampler (ASMC, [2]_); then  :math:`p(\mathbf{q}_{1:n_D}\,|\,\mathbf{y}, n_D, \mathbf{r}_{1:n_D})` is analytically computed.

Adaptive Sequential Monte Carlo sampler for the marginal posterior :math:`p(n_D, \mathbf{r}_{1:n_D}\,|\,\mathbf{y})`.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
| ASMC aims at approximating the target pdf :math:`p(n_D, \mathbf{r}_{1:n_D}\,|\,\mathbf{y})` using a large set of samples, also called particles; in our context each particle is a candidate solution, i.e. the number of dipoles and the dipole locations.
| One easy way to produce such set of samples is to draw them independently from a simple pdf, and possibly weigh them to correctly approximate the target pdf (Importance Sampling, IS [3]_). An alternative approach is to start from a random candidate, perturb it randomly many times, and then approximate the target pdf with the collection of samples along the iterations (Markov Chain Monte Carlo, MCMC [3]_).
| ASMC combines these two techniques:  multiple samples are independently drawn from a simple distribution, in our case the prior pdf :math:`p(n_D, \mathbf{r}_{1:n_D})`, evolve following an MCMC scheme, and their weights are updated after every MCMC step; at times, a resample move is performed, that means samples having negligible weights are replaced by samples in the higher--probability region, so as to explore better these areas. Eventually, the target distribution is approximated by the weighted sample set obtained at the last iteration.

Analytic computation of :math:`p(\mathbf{q}_{1:n_D}\,|\,\mathbf{y}, n_D, \mathbf{r}_{1:n_D})`.
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
By exploiting the mutual independence of :math:`\mathbf{J}` and :math:`\mathbf{N}` and the Gaussian assumptions made about the prior pdf of the dipole moments and the noise model, SESAME analytically compute the posterior pdf :math:`p(\mathbf{q}_{1:n_D}\,|\,\mathbf{y}, n_D, \mathbf{r}_{1:n_D})`. Indeed, it is a Gaussian density whose mean and variance depend only on the data, the forward solution :math:`\mathbf{G}\left(\mathbf{r}_{1:n_D}\right)`, and the standard deviations :math:`\sigma_q` and :math:`\sigma_e`.

Get the most out of Sesame's results.
-------------------------------------
Sesame's output consists in a set of weighted particles

.. math::  \left\{\big(n_D^{i}, \mathbf{r}_{1:n_D^{i}}^{i}\big), w^{i} \right\}_{i=1, ..., I}

which allows to approximate the full posterior distribution :math:`p(\mathbf{j}|\mathbf{y})`. |br|
Roughly speaking, each of the :math:`I` particles represents a candidate source configuration, while the weight :math:`w^i` quantifies how likely it is that such configuration may have generated the recorded data. |br|
Sesame also allows to compute the most probable source configuration through the following procedure. |br|
First the most probable model is identified by estimating mode of the posterior pdf for the number of sources i.e. 

.. math:: \hat{n}_D = argmax\, p \left(n_D | \mathbf{y} \right) = argmax\, \sum_{i=1}^I w^{i} \delta \left(n_D-n_D^{i} \right).

Subsequently, for each point :math:`r` of the cortical discretization, we compute

.. math:: p(r| \mathbf{y},\hat{n}_D) = \sum_{i=1}^I w^i \delta\left(\hat{n}_D,n_D^i\right) \sum_{k=1}^{n_D^{i}} \delta\left(r, r_k^{i}\right)\, ,

which represents the posterior probability of a source being located in :math:`r`. This quantity can be used to produce posterior maps of activation on the cortical surface and to compute estimates of dipole locations as the local peaks of such a probability map. |br|
Finally, dipole moments can be reasonably estimated as the mean of the corresponding Gaussian distribution.


Reference
---------
.. [1] S. Sommariva and A. Sorrentino, `Sequential Monte Carlo samplers for semi-linear inverse problems and application to Magnetoencephalography <https://doi.org/10.1088/0266-5611/30/11/114020>`_. Inverse Problems, 30 114020 (2014).
.. [2] A. Sorrentino, G. Luria, and R. Aramini, `Bayesian multi-dipole modeling of a single topography in MEG by adaptive Sequential Monte Carlo Samplers <https://iopscience.iop.org/article/10.1088/0266-5611/30/4/045010>`_. Inverse Problems, 30 045010 (2014).
.. [3] C. Robert and G. Casella, `Monte Carlo Statistical Methods <https://www.springer.com/gp/book/9780387212395>`_, 2nd Edition.  Springer (2004).
