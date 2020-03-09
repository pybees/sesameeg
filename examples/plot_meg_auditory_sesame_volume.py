"""
=======================================================
Compute SESAME inverse solution in volume source space
=======================================================

Compute and visualize SESAME solution on the auditory sample dataset
in a volume source space.
"""
# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#          Albero Sorrentino <sorrentino@dima.unige.it>
#
# License: BSD (3-clause)

from os import path as op
import matplotlib.pyplot as plt
import numpy as np
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img
from mne import read_forward_solution, pick_types_forward, read_evokeds, \
    read_trans, head_to_mri
from mne.datasets import sample
from mne.label import _n_colors
from sesameeg import Sesame
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
# Load the transformation matrix, the forward solution :math:`\textbf{G}` and the evoked data
# :math:`\textbf{y}`.
# The forward solution also defines the employed brain discretization.

# Transformation matrix
trans = read_trans(fname_trans)

# Choose sensor type
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
n_parts = 100
# If None, sigma_noise and sigma_q will be estimated by SESAME.
sigma_noise = None
sigma_q = None

cov = None
# If a noise covariance is provided, SESAME will pre-whiten the data
# from mne import read_cov
# fname_cov = op.join(sample.data_path(), 'MEG', subject,
#                    'sample_audvis-cov.fif')
# cov = read_cov(fname_cov)


_sesame = Sesame(fwd, evoked, n_parts=n_parts, s_noise=sigma_noise,
                 sample_min=sample_min, sample_max=sample_max,
                 s_q=sigma_q, hyper_q=True, cov=cov, subsample=subsample,
                 verbose=True)
time_start = time.time()
_sesame.apply_sesame()
time_elapsed = (time.time() - time_start)
print('    Estimated number of sources: {0}'.format(_sesame.est_n_dips[-1]))
print('    Estimated source locations: {0}'.format(_sesame.est_locs[-1]))
print('    Total computation time: {0}'.format(time_elapsed))

# Compute goodness of fit
gof = _sesame.goodness_of_fit()
print('    Goodness of fit with the recorded data: {0}%'.format(round(gof, 4) * 100))

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
stc = _sesame.compute_stc(subject)

peak_vertex, peak_time = stc.get_peak(vert_as_index=True,
                                      time_as_index=True)
peak_pos = fwd['source_rr'][peak_vertex]
peak_mri_pos = head_to_mri(peak_pos, mri_head_t=trans,
                           subject=subject, subjects_dir=subjects_dir)

_time = stc.times[-1]
stc.crop(_time-1, _time)

img = stc.as_volume(fwd['src'], mri_resolution=True)
plot_stat_map(index_img(img, -1), fname_t1, threshold=0.001, cut_coords=peak_mri_pos)
plt.show()

########################################################################################
# You can save result in HDF5 files with:
# _sesame.save_h5(save_fname, tmin=time_in, tmax=time_fin, subsample=subsample,
#                 sbj=subject, data_path=fname_evoked, fwd_path=fname_fwd)

# You can save result in Pickle files with:
# _sesame.save_pkl(save_fname, tmin=time_in, tmax=time_fin, subsample=subsample,
#                  sbj=subject, data_path=fname_evoked, fwd_path=fname_fwd)

