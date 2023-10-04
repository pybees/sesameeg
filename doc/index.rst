.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SESAMEEG
========

SESAMEEG is a Python3 library providing the Bayesian multi-dipole localization method
SESAME [1]_ (SEquential Semi-Analytic Montecarlo Estimator) for the automatic estimation of
brain source currents from MEEG data, either in the time domain and in the frequency domain [2]_.

A mathematical description of the algorithm is available in the
:doc:`documentation <explanation_algorithm>`.


Installation
============

To install this package, the easiest way is using ``pip``. It will install both
``sesameeg`` and its dependencies. The ``setup.py`` depends on ``numpy``, ``scipy``  and ``mne``
for the installation so it is advised to install them beforehand.
To install this package, please run the following commands:

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
.. [2] G. Luria et al., `Bayesian multi-dipole modelling in the frequency domain <https://doi.org/10.1016/j.jneumeth.2018.11.007>`_. J. Neurosci. Meth., 312 (2019) 27–36.

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






