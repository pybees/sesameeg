=============
API Reference
=============

This is the reference for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of Sesameeg.




Main class
==========
.. currentmodule:: sesameeg

.. autosummary::
   :toctree: generated/
   :template: class.rst

    Sesame


MNE-Python utility functions
============================
.. currentmodule:: sesameeg.mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

    prepare_sesame

Visualization
=============
.. currentmodule:: sesameeg.viz

.. autosummary::
   :toctree: generated/
   :template: function.rst

    plot_n_sources
    plot_stc
    plot_vol_stc
    plot_cloud_sources
    plot_amplitudes

Reading SESAME result
=====================
.. currentmodule:: sesameeg.io

.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_h5

Other classes
=============
.. currentmodule:: sesameeg

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Dipole
   Particle
   EmpPdf

Utility functions
=================
.. currentmodule:: sesameeg.utils

.. autosummary::
   :toctree: generated/
   :template: function.rst

   prior_loc_from_labels
   compute_neighbours_matrix
   compute_neighbours_probability_matrix
   estimate_dip_mom_std
   estimate_noise_std
   initialize_radius
