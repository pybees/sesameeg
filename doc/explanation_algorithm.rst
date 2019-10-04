############
Explanation
############

Spiegazione della section.

Input
-----

SESAME in action
----------------

Output
------

Multi--dipole source model.
----------------------------
We model the neural sources producing the recorded MEG/EEG data using a primary current distribution which is approximated by the superposition of an unknown number of current dipoles.
In mathematical terms a source configuration is assumed to belong to the variable--dimension state--space model

.. math::  \mathcal{J}\ :=\ \bigcup_{n_D}\ \{n_D\} \times \mathcal{D}^{n_D} ,
   :label: model
    
were :math:`n_D` is the number of dipoles, :math:`\mathcal{D}` is the state space for a single--dipole configuration and :math:`\mathcal{D}^{n_D}` is the Cartesian product of :math:`n_D` copies of :math:`\mathcal{D}`.
Esempio referenza :eq:`model`
