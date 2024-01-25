import inspect
import numpy as np
import os.path as op
import pyvista as pv
import matplotlib.pyplot as plt
from mne import head_to_mri
from mne.label import _n_colors
from mne.viz.backends._utils import _qt_app_exec
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img
from pyvistaqt import BackgroundPlotter


def plot_n_sources(inv_op, kind='bar', title=None):
    """
    Plot the probability of number of sources.

    Parameters
    ----------
    inv_op : Instance of :py:class:`~sesameeg.Sesame`
        Instance of `Sesame`.
    kind : :py:class:`~str`
        The kind of plot to draw. Options are: “bar”, “pie”.
    title : :py:class:`~str` | None
        Figure title.
    """
    _model_sel = inv_op.model_sel[-1]
    _labels = np.array(
        ['no sources', 'one source', 'two sources', 'three sources', 'four sources', 'five sources',
         'six sources', 'seven sources', 'eight sources', 'nine sources', 'ten sources'])
    _colors = np.asarray(plt.colormaps['tab10'].resampled(inv_op.max_n_dips).colors)
    if kind == 'bar':
        fig = plt.figure(title, figsize=(10, 5))

        n_src = np.arange(inv_op.max_n_dips + 1)
        bar_labels = n_src

        plt.bar(list(n_src), _model_sel, label=bar_labels, color=_colors)

        plt.ylim((0, 1))
        plt.xticks(np.arange(0, inv_op.max_n_dips + 1, 1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.grid(axis='y')
        fig.axes[0].set_axisbelow(True)

        plt.xlabel('Number of sources')
        plt.ylabel('Probability')
        plt.title('Marginal probability for the number of sources')

        plt.show()
    elif kind == 'pie':
        _ms = _model_sel[np.nonzero(_model_sel)]
        _lbl = _labels[np.nonzero(_model_sel)]
        _clr = _colors[np.nonzero(_model_sel)]
        _explode = [0 for i in _ms]
        _explode[np.argmax(_ms)] += 0.1

        fig = plt.figure(title)
        plt.pie(_ms, labels=_lbl, colors=_clr, autopct='%1.1f%%', explode=_explode, shadow=True)
        plt.title('Marginal probability for the number of sources')
        plt.show()
    else:
        raise ValueError("Parameter 'kind' may be set either to 'bar' or to 'pie'.")


def plot_amplitudes(inv_op, title=None):
    """
    Plot the amplitude of the estimated sources as function of time.

    Parameters
    ----------
    inv_op : Instance of :py:class:`~sesameeg.Sesame`
        Instance of `Sesame`.
    title : :py:class:`~str` | None
        Figure title.
    """
    est_n_dips = inv_op.est_n_dips[-1]
    if inv_op.fixed_ori:
        amplitude = np.abs(inv_op.est_dip_moms.T)
    else:
        amplitude = np.array([np.linalg.norm(inv_op.est_dip_moms[:, 3 * i_d:3 * (i_d + 1)],
                                             axis=1) for i_d in range(est_n_dips)])
    colors = _n_colors(est_n_dips)
    plt.figure(title)
    plt.clf()
    for idx, amp in enumerate(amplitude):
        if inv_op.data_times is not None:
            plt.plot(1e3 * inv_op.data_times, 1e9 * amp, color=colors[idx], linewidth=2)
            plt.xlabel('Time (ms)')
        else:
            plt.plot(1e9 * amp, color=colors[idx], linewidth=2)
    plt.ylabel('Source amplitude (nAm)')
    plt.show()


def plot_stc(inv_op, plot_kwargs=None, savepath=None,
             save_kwargs=None, true_idxs=None, force_open=False):
    """
    Plot SESAME source estimates using mne.

    Parameters
    ----------
    inv_op : Instance of :py:class:`~sesameeg.Sesame`
        Instance of `Sesame`. Source estimates have to be computed through :py:meth:`~sesameeg.Sesame.compute_stc`
    plot_kwargs : :py:class:`~dict` | None
        Additional arguments to :py:func:`~mne.viz.plot_source_estimates` (e.g., dict(surface='white')).
    savepath : :py:class:`~str` | None
        File path to write image to. If None, no image is written.
    save_kwargs : :py:class:`~dict` | None
        Additional arguments to :py:meth:`~pyvista.Plotter.screenshot`.
    true_idxs : :py:class:`~numpy.ndarray` | None
        In simulation settings, indexes of source space points in which true sources are located.
    force_open : :py:class:`~bool`
        If True, force the image to stay open.
    """
    if inv_op.stc is None:
        raise AttributeError('Compute Sesame stc first.')
    est_n_dips = inv_op.est_n_dips[-1]
    est_locs = inv_op.est_locs[-1]
    colors = _n_colors(est_n_dips)
    brain_kwargs = {'surface': 'inflated',
                    'hemi': 'split',
                    'clim': {'kind': 'value', 'lims': [1e-4, 1e-1, 1]},
                    'time_label': ' ',
                    'subjects_dir': inv_op.subjects_dir,
                    'size': (1000, 600),
                    'show_traces': False}
    view_kwargs = dict()

    if plot_kwargs is not None:
        _show_view_keys = ['roll', 'distance', 'row', 'col', 'align', 'azimuth',
                           'elevation', 'focalpoint']
        _valid_keys = inspect.signature(inv_op.stc.plot).parameters.keys()
        aux_dict = dict()
        for _k, _v in plot_kwargs.items():
            if _k in _show_view_keys:
                view_kwargs[_k] = _v
            else:
                if _k not in _valid_keys:
                    print(f'WARNING! Removing invalid stc.plot keyword : {_k}')
                else:
                    aux_dict[_k] = _v
        brain_kwargs.update(aux_dict)

    brain = inv_op.stc.plot(inv_op.subject, **brain_kwargs,
                            brain_kwargs=dict(show=False))
    nv_lh = inv_op.stc.vertices[0].shape[0]

    if true_idxs is not None:
        print(true_idxs)
        for t_idx, t_loc in enumerate(true_idxs):
            if t_loc < nv_lh:
                brain.add_foci(inv_op.stc.vertices[0][t_loc], coords_as_verts=True,
                               hemi='lh', color='k', scale_factor=0.3)
            else:
                brain.add_foci(inv_op.stc.vertices[1][t_loc - nv_lh], coords_as_verts=True,
                               hemi='rh', color='k', scale_factor=0.3)

    for idx, loc in enumerate(est_locs):
        if loc < nv_lh:
            brain.add_foci(inv_op.stc.vertices[0][loc], coords_as_verts=True,
                           hemi='lh', color=colors[idx], scale_factor=0.3)
        else:
            brain.add_foci(inv_op.stc.vertices[1][loc - nv_lh], coords_as_verts=True,
                           hemi='rh', color=colors[idx], scale_factor=0.3)

    if len(view_kwargs) > 0:
        brain.show_view(**view_kwargs)

    brain.show()

    if force_open:
        _qt_app_exec(brain._renderer.figure.store["app"])

    if savepath is not None:
        s_kw = dict(filename=savepath)
        if save_kwargs is not None:
            _valid_keys = inspect.signature(pv.Plotter.screenshot).parameters.keys()
            aux_dict = dict()
            for _k, _v in save_kwargs.items():
                if _k not in _valid_keys:
                    print(f'WARNING! Removing invalid pyvista.Plotter.screenshot keyword : {_k}')
                else:
                    aux_dict[_k] = _v
            s_kw.update(aux_dict)
        brain.plotter.screenshot(**s_kw)


def plot_vol_stc(inv_op, plot_kwargs=None, savepath=None, save_kwargs=None):
    """
    Plot Nutmeg style SESAME volumetric source estimates using nilearn.

    Parameters
    ----------
    inv_op : Instance of :py:class:`~sesameeg.Sesame`
        Instance of `Sesame`.
        Source estimates have to be computed through :py:meth:`~sesameeg.Sesame.compute_stc`
    plot_kwargs : :py:class:`~dict` | None
        Additional arguments to :py:func:`~nilearn.plotting.plot_stat_map` (e.g., dict(colorbar=False)).
    savepath : :py:class:`~str` | None
        File path to write image to. If None, no image is written.
    save_kwargs : :py:class:`~dict` | None
        Additional arguments to :py:func:`~matplotlib.pyplot.savefig`.
    """

    if inv_op.stc is None:
        raise AttributeError('Compute Sesame stc first.')

    peak_vertex, peak_time = inv_op.stc.get_peak(vert_as_index=True,
                                          time_as_index=True)
    peak_pos = inv_op.forward['source_rr'][peak_vertex]
    peak_mri_pos = head_to_mri(peak_pos, mri_head_t=inv_op.trans_matrix, kind='ras',
                               subject=inv_op.subject, subjects_dir=inv_op.subjects_dir)

    _time = inv_op.stc.times[-1]
    inv_op.stc.crop(_time - 1, _time)

    img = inv_op.stc.as_volume(inv_op.forward['src'], mri_resolution=True)
    fig_vol = plt.figure(inv_op.subject, figsize=(12, 4))

    _mri_fpath = op.join(inv_op.subjects_dir, inv_op.subject, 'mri', 'T1.mgz')
    if op.isfile(_mri_fpath):
        bg_img = _mri_fpath
    else:
        bg_img = None
    brain_kwargs = dict(bg_img=bg_img, threshold=0.001, cut_coords=peak_mri_pos, figure=fig_vol)

    if plot_kwargs is not None:
        _valid_keys = inspect.signature(plot_stat_map).parameters.keys()
        aux_dict = dict()
        for _k, _v in plot_kwargs.items():
            if _k not in _valid_keys:
                print(f'WARNING! Removing invalid plot_stat_map keyword : {_k}')
            else:
                aux_dict[_k] = _v
        brain_kwargs.update(aux_dict)

    plot_stat_map(index_img(img, -1), **brain_kwargs)

    if savepath is not None:
        s_kw = dict(fname=savepath, dpi=100)
        if save_kwargs is not None:
            _valid_keys = ['dpi', 'format', 'metadata', 'bbox_inches', 'pad_inches',
                           'facecolor', 'edgecolor', 'backend']
            aux_dict = dict()
            for _k, _v in save_kwargs.items():
                if _k not in _valid_keys:
                    print(f'WARNING! Removing invalid matplotlib.pyplot.savefig keyword : {_k}')
                else:
                    aux_dict[_k] = _v
            s_kw.update(aux_dict)
        plt.savefig(**s_kw)
    plt.show()


def plot_cloud_sources(inv_op, savepath=None, true_idxs=None):
    """
    Plot point cloud style SESAME source estimates using pyvista.

    Parameters
    ----------
    inv_op : Instance of :py:class:`~sesameeg.Sesame`
        Instance of `Sesame`. Source estimates have to be computed through :py:meth:`~Sesame.compute_stc`
    savepath : :py:class:`~str` | None
        File path to write image to. If None, no image is written.
    true_idxs : :py:class:`~numpy.ndarray` | None
        In simulation settings, indexes of source space points in which true sources are located.
    """
    est_n_dips = inv_op.est_n_dips[-1]
    est_locs = inv_op.est_locs[-1]
    pmap = inv_op.pmap[0].sum(axis=0)
    colors = _n_colors(est_n_dips)
    cloud = pv.PolyData(inv_op.source_space)
    blob = pv.PolyData(inv_op.source_space[np.nonzero(pmap)[0]])
    blob['probability'] = pmap[np.nonzero(pmap)[0]]

    plotter = BackgroundPlotter()
    plotter.background_color = "white"
    plotter.add_mesh(cloud, color='black', point_size=5.0, render_points_as_spheres=True)
    plotter.add_mesh(blob, scalars='probability', point_size=10.0, render_points_as_spheres=True,
                     cmap='viridis_r', clim=(1e-4, 1),
                     scalar_bar_args={'title': "Probability", 'color': 'black'}
                     )

    if true_idxs is not None:
        for t_idx, t_loc in enumerate(true_idxs):
            plotter.add_mesh(inv_op.source_space[t_loc], color='black', point_size=20.0,
                             render_points_as_spheres=True)

    for idx, loc in enumerate(est_locs):
        plotter.add_mesh(inv_op.source_space[loc], color=colors[idx], point_size=20.0,
                         render_points_as_spheres=True)
    if savepath is not None:
        plotter.screenshot(savepath)
    plotter.show()

