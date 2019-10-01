.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Sesame
========================================

This is a python implementation of the Bayesian multi-dipole modeling method and Sequential Monte Carlo algorithm described in [1]_ and currently implemented in BESA Research 7.0.

The algorithm takes in input a sourcespace, a leadfield and a data time series, and outputs a posterior probability map for source locations, the estimated number of dipoles, their locations and their amplitudes.

Authors of the code
-------------------
| Gianvittorio Luria <luria@dima.unige.it>,
| Sara Sommariva <sommariva@dima.unige.it>,
| Alberto Sorrentino <sorrentino@dima.unige.it>.

References
----------
.. [1] `S. Sommariva and A. Sorrentino, Sequential Monte Carlo samplers for semi-linear inverse problems and application to Magnetoencephalography. Inverse Problems, 30 114020 (2014).   Problems 30(11):114020 (2014) <https://doi.org/10.1088/0266-5611/30/11/114020>`_

.. toctree::
   :maxdepth: 2
   :hidden:
   
   auto_examples/index
   api
   explanation_algorithm
   
.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Old stuff

   quick_start
   user_guide






