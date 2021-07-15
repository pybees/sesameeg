.. -*- mode: rst -*-

SESAMEEG: SEquential Semi-Analytic Montecarlo Estimation for MEEG
=================================================================

This is a Python3 implementation of the Bayesian multi-dipole modeling method and Sequential Monte Carlo algorithm
SESAME described in [1]_.
The algorithm takes in input a forward solution and a MEEG evoked data time series, and outputs a posterior
probability map for brain activity, as well as estimates of the  number of sources, their locations and their
amplitudes.

Installation
============

To install this package, the easiest way is using ``pip``. It will install this
package and its dependencies. The ``setup.py`` depends on ``numpy``, ``scipy``  and ``mne``
for the installation so it is advised to install them beforehand. To
install this package, please run the following commands:

(Latest stable version)

.. code::

    pip install numpy scipy mne
    pip install sesameeg

If you do not have admin privileges on the computer, use the ``--user`` flag
with ``pip``. To upgrade, use the ``--upgrade`` flag provided by ``pip``.

To check if everything worked fine, you can run:

.. code::

	python -c 'import sesameeg'

and it should not give any error messages.


Bug reports
===========

Use the `github issue tracker <https://github.com/pybees/sesameeg/issues>`_ to report bugs.


Authors of the code
-------------------
| Gianvittorio Luria <luria@dima.unige.it>,
| Sara Sommariva <sommariva@dima.unige.it>,
| Alberto Sorrentino <sorrentino@dima.unige.it>.

Cite our work
=============

If you use this code in your project, please consider citing our work:

.. [1] S. Sommariva and A. Sorrentino, `Sequential Monte Carlo samplers for semi-linear inverse problems and application to Magnetoencephalography <https://doi.org/10.1088/0266-5611/30/11/114020>`_. Inverse Problems, 30 114020 (2014).
