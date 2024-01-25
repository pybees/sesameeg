"""
=====================================================================
Compute SESAME inverse solution on evoked data in volume source space
=====================================================================

In this example we shall compute SESAME inverse solution on evoked data in
a volume source space. Data are taken from the MNE-Python
`sample <https://mne.tools/stable/generated/mne.datasets.sample.data_path.html#mne.datasets.sample.data_path>`_
dataset and correspond to the response to an auditory stimulus.
"""
# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#          Alberto Sorrentino <sorrentino@dima.unige.it>
#
# License: BSD (3-clause)

# sphinx_gallery_thumbnail_number = 3

from os import path as op
import matplotlib.pyplot as plt
import numpy as np
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img
from mne import read_forward_solution, pick_types_forward, read_evokeds, \
    read_trans, head_to_mri
from mne.datasets import sample
from mne.label import _n_colors
from sesameeg.mne import prepare_sesame
import time


data_path = sample.data_path()
subject = 'sample'
subjects_dir = op.join(data_path, 'subjects')
fname_fwd = op.join(data_path, 'MEG', subject,
                    'sample_audvis-meg-vol-7-fwd.fif')
fname_trans = op.join(data_path, 'MEG', subject,
                      'sample_audvis_raw-trans.fif')
fname_t1 = op.join(data_path , 'subjects', subject, 'mri', 'T1.mgz')
fname_evoked = op.join(data_path, 'MEG', subject, 'sample_audvis-ave.fif')

###############################################################################
# Load the  mri-to-head coordinates transformation matrix, the forward solution
# :math:`\textbf{G}` and the evoked data :math:`\textbf{y}`.
# The forward solution also defines the employed brain discretization which, in this example,
# comprises the whole brain volume.

# Transformation matrix
trans = read_trans(fname_trans)

# Choose sensor type
meg_sensor_type = True  # All meg sensors will be included
eeg_sensor_type = False

# Forward solution
fwd = read_forward_solution(fname_fwd, exclude='bads')
fwd = pick_types_forward(fwd, meg=meg_sensor_type, eeg=eeg_sensor_type, ref_meg=False)

# Evoked Data
condition = 'Left Auditory'
evoked = read_evokeds(fname_evoked, condition=condition, baseline=(None, 0))
evoked = evoked.pick('meg', exclude='bads')

###############################################################################
# Define the parameters.
time_min, time_max = 0.045, 0.135  # Select N100m
subsample = None
sample_min, sample_max = evoked.time_as_index([time_min, time_max], use_rounding=True)

# To accelerate the run time of this example, we use a small number of
# particles. We recall that the parameter ``n_parts`` represents, roughly speaking,
# the number of candidate solutions that are tested in the Monte Carlo procedure;
# larger values yield in principle more accurate reconstructions but also entail a
# higher computational cost. Setting the value to about a hundred seems to represent
# a good trade–off.
n_parts = 10
# If None, noise_std and dip_mom_std will be estimated by SESAME.
noise_std = None
dip_mom_std = None
noise_cov = None

# You can make SESAME pre-whiten the data by providing a noise covariance
# from mne import read_cov
# fname_cov = op.join(sample.data_path(), 'MEG', subject,
#                    'sample_audvis-cov.fif')
# noise_cov = read_cov(fname_cov)

###############################################################################
# Visualize the selected data.

lst = evoked.plot_joint(show=False)
for fig in lst:
    ax = fig.get_axes()
    ax[0].axvline(time_min, color='r', linewidth=2.0)
    ax[0].axvline(time_max, color='r', linewidth=2.0)
plt.show()

###############################################################################
# Apply SESAME.
_sesame = prepare_sesame(fwd, evoked, n_parts=n_parts, noise_std=noise_std,
                         top_min=time_min, top_max=time_max, dip_mom_std=dip_mom_std,
                         hyper_q=True, noise_cov=noise_cov, subsample=subsample,
                         subject=subject, subjects_dir=subjects_dir, trans_matrix=trans)
time_start = time.time()
_sesame.apply_sesame()
time_elapsed = (time.time() - time_start)
print(f'    Total computation time: {round(time_elapsed, 2)} seconds.')

# Compute goodness of fit
gof = _sesame.goodness_of_fit()
print('    Goodness of fit with the recorded data: {0}%'.format(round(gof, 4) * 100))

###############################################################################
# Visualize the posterior map of the dipoles' location
# :math:`p(r| \textbf{y}, 2)` as an overlay onto the MRI.
_sesame.plot_sources(plot_kwargs={'distance': 650})

###############################################################################
# Visualize amplitude of the estimated sources as function of time.
_sesame.plot_source_amplitudes()

#######################################################################################
# Save results.

# You can save SESAME result in an HDF5 file with:
# _sesame.save_h5(save_fname, sbj=subject, data_path=fname_evoked, fwd_path=fname_fwd)

# You can save SESAME result in a Pickle file with:
# _sesame.save_pkl(save_fname, sbj=subject, data_path=fname_evoked, fwd_path=fname_fwd)
