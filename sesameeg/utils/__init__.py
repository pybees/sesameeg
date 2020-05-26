# # # WARNING # # #
# This list must also be updated in doc/_templates/autosummary/class.rst if it
# is changed here!

from .smc_utils import (compute_neighbours_matrix, compute_neighbours_probability_matrix,
                        initialize_radius, estimate_noise_std, estimate_dip_mom_std,
                        gamma_pdf, is_epochs, is_evoked, is_forward, sample_from_sphere, woodbury)

from .check import _check_h5_installed, _check_pickle_installed
