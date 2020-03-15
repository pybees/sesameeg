====================
Python API Reference
====================

This is the reference for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of Sesameeg.

.. currentmodule:: sesameeg


Main classes
============

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Sesame

Other classes
=============

.. autosummary::
   :toctree: generated/
   :template: class.rst

   dipoles.Dipole
   particles.Particle
   emp_pdf.EmpPdf

Utility functions
=================
.. currentmodule:: sesameeg.utils

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_neighbours_matrix
   compute_neighbours_probability_matrix
   estimate_s_q
   estimate_s_noise
   initialize_radius

Reading SESAME result
=====================

.. currentmodule:: sesameeg.io

.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_h5

