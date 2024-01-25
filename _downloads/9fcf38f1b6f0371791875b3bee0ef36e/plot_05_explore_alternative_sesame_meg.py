"""
===========================================================
Explore SESAME alternative inverse solutions on evoked data
===========================================================

In this example we shall show that SESAME is able to provide alternative solutions
with non-negligible probabilities and how the user can easily explore all of them.

To do so, we shall once again apply SESAME on an evoked dataset,
corresponding to the response to an auditory stimulus. Data are taken from the MNE-Python
`sample <https://mne.tools/stable/generated/mne.datasets.sample.data_path.html#mne.datasets.sample.data_path>`_
dataset.
"""

# Author: Gianvittorio Luria <luria@dima.unige.it>
#
# License: BSD (3-clause)

# sphinx_gallery_thumbnail_number = 2

from os import path as op
import numpy as np
import matplotlib.pyplot as plt

from mne.datasets import sample
from mne import read_forward_solution, pick_types_forward, read_evokeds

from sesameeg.mne import prepare_sesame

data_path = sample.data_path()
subject = 'sample'
subjects_dir = op.join(data_path, 'subjects')
fname_fwd = op.join(data_path, 'MEG', subject,
                    'sample_audvis-meg-eeg-oct-6-fwd.fif')
fname_evoked = op.join(data_path, 'MEG', subject, 'sample_audvis-ave.fif')

###############################################################################
# Load the forward solution  :math:`\textbf{G}`  and the evoked data
# :math:`\textbf{y}`.
# The forward solution also defines the employed brain discretization.
meg_sensor_type = True  # All MEG sensors will be included
eeg_sensor_type = False

# Forward solution
fwd = read_forward_solution(fname_fwd, exclude='bads')
fwd = pick_types_forward(fwd, meg=meg_sensor_type,
                         eeg=eeg_sensor_type, ref_meg=False)

# Evoked Data
condition = 'Left Auditory'
evoked = read_evokeds(fname_evoked, condition=condition, baseline=(None, 0))
evoked = evoked.pick('meg', exclude='bads')

###############################################################################
# Define the parameters.
time_min, time_max = 0.045, 0.135  # Select N100m
sample_min, sample_max = evoked.time_as_index([time_min, time_max],
                                              use_rounding=True)

# Fix the random seed
np.random.seed(3)

# Manually set the noise standard deviation value. We are intentionally
# overestimating the parameter with respect to SESAME’s default.
noise_std = 0.64 * np.max(abs(evoked.data))

# If None, dip_mom_std will be estimated by SESAME.
dip_mom_std = None


###############################################################################
# Visualize the selected data.
fig = evoked.plot(show=False)
for ax in fig.get_axes()[:2]:
    ax.axvspan(time_min, time_max, alpha=0.3, color="#66CCEE")
plt.show()

###############################################################################
# Apply SESAME.
_sesame = prepare_sesame(fwd, evoked, noise_std=noise_std,
                         top_min=time_min, top_max=time_max,
                         dip_mom_std=dip_mom_std, hyper_q=True,
                         subject=subject, subjects_dir=subjects_dir)
_sesame.apply_sesame()


# Compute goodness of fit
gof = _sesame.goodness_of_fit()
print('    Goodness of fit with the recorded data: {0}%'.format(round(gof, 4) * 100))

# Compute source dispersion
sd = _sesame.source_dispersion()
print('    Source Dispersion: {0} mm'.format(round(sd, 2)))

###############################################################################
# Visualize the marginal probability of the number of sources
_sesame.plot_source_number(kind='pie')

###############################################################################
# Visualize the posterior map of the dipoles' location
# :math:`p(r| \textbf{y}, 2)` and the estimated sources on the inflated brain.
_sesame.plot_sources(plot_kwargs={'distance': 650})

###############################################################################
# Visualize the amplitude of the estimated sources as function of time.
_sesame.plot_source_amplitudes()

###############################################################################
# Explore the alternative scenario
_sesame.plot_sources(n_sources=2, plot_kwargs={'distance': 650})
_sesame.plot_source_amplitudes(n_sources=2)
