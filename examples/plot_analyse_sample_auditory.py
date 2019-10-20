"""
=============================
Compute SESAME on evoked data
=============================

Compute and visualize SESAME solution on the auditory sample dataset.
"""
# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#          Albero Sorrentino <sorrentino@dima.unige.it>
#
# License: BSD (3-clause)

from os import path as op
import numpy as np
import matplotlib.pyplot as plt

from mne.datasets import sample
from mne import read_forward_solution, pick_types_forward, read_evokeds
from mne.label import _n_colors

from sesame.sasmc import Sesame
from mayavi import mlab

data_path = sample.data_path()
subject = 'sample'
subjects_dir = op.join(data_path, 'subjects')
fname_fwd = op.join(data_path, 'MEG', subject,
                    'sample_audvis-meg-eeg-oct-6-fwd.fif')
fname_evoked = op.join(data_path, 'MEG', subject, 'sample_audvis-ave.fif')

###############################################################################
# Load the forward solution :math:`\textbf{G}` and the evoked data
# :math:`\textbf{y}`.
# The forward solution also defines the employed brain discretization.
meg_sensor_type = True  # All meg sensors will be included
eeg_sensor_type = False

# Forward solution
fwd = read_forward_solution(fname_fwd, exclude='bads')
fwd = pick_types_forward(fwd, meg=meg_sensor_type,
                         eeg=eeg_sensor_type, ref_meg=False)

# Evoked Data
condition = 'Left Auditory'
evoked = read_evokeds(fname_evoked, condition=condition, baseline=(None, 0))
evoked = evoked.pick_types(meg=meg_sensor_type,
                           eeg=eeg_sensor_type, exclude='bads')

###############################################################################
# Select and visualize the data topographies.
time_in = 0.055
time_fin = 0.135
subsample = None
sample_min, sample_max = evoked.time_as_index([time_in, time_fin],
                                              use_rounding=True)
fig = evoked.plot(show=False)
for ax in fig.get_axes():
    ax.axvline(time_in, color='r', linewidth=2.0)
    ax.axvline(time_fin, color='r', linewidth=2.0)
plt.show()

###############################################################################
# Run SESAME.
n_parts = 10
# If None, sigma_noise and sigma_q will be estimated by SESAME.
sigma_noise = None
sigma_q = None

cov = None
# If a noise covariance is provided, SESAME will be pre-whiten the data
# from mne import read_cov
# fname_cov = op.join(sample.data_path(), 'MEG', subject,
#                    'sample_audvis-cov.fif')
# cov = read_cov(fname_cov)


_sesame = Sesame(fwd, evoked, n_parts=n_parts, s_noise=sigma_noise,
                 sample_min=sample_min, sample_max=sample_max,
                 s_q=sigma_q, cov=cov, subsample=subsample, verbose=False)
_sesame.apply_sesame()
gof = _sesame.goodness_of_fit()

print('    Estimated number of sources: {0}'.format(_sesame.est_n_dips[-1]))
print('    Estimated source locations: {0}'.format(_sesame.est_locs[-1]))
print('    Goodness of fit with the recorded data: {0}'.format(gof))

###############################################################################
# Visualize amplitude of the estimated sources as function of time.
est_n_dips = _sesame.est_n_dips[-1]
est_locs = _sesame.est_locs[-1]

times = evoked.times[_sesame.s_min:_sesame.s_max+1]
amplitude = np.array([np.linalg.norm(_sesame.est_q[:, i_d:3 * (i_d + 1)],
                                     axis=1) for i_d in range(est_n_dips)])
colors = _n_colors(est_n_dips)
plt.figure()
for idx, amp in enumerate(amplitude):
    plt.plot(times, 1e9*amp, color=colors[idx], linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Source amplitude (nAm)')
plt.show()

###############################################################################
# Visualize the posterior map of the dipoles' location
# :math:`p(r| \textbf{y}, 2)` and the estimated sources on the inflated brain.
stc = _sesame.to_stc(subject)
clim = dict(kind='value', lims=[1e-4, 1e-1, 1])
brain = stc.plot(subject, surface='inflated', hemi='split', clim=clim,
                 time_label=' ', subjects_dir=subjects_dir, size=(1000, 600))
nv_lh = stc.vertices[0].shape[0]
for idx, loc in enumerate(est_locs):
    if loc < nv_lh:
        brain.add_foci(stc.vertices[0][loc], coords_as_verts=True,
                       hemi='lh', color=colors[idx], scale_factor=0.3)
    else:
        brain.add_foci(stc.vertices[1][loc-nv_lh], coords_as_verts=True,
                       hemi='rh', color=colors[idx], scale_factor=0.3)

mlab.show()
