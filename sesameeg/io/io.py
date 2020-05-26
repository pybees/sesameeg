import numpy as np
from mne import SourceEstimate, VolSourceEstimate
from ..utils import _check_h5_installed, _check_pickle_installed


def _export_to_stc(inv_op, subject=None):
    if not hasattr(inv_op, 'pmap'):
        raise AttributeError('Run the inversion algorithm first!!')

    pmaps = inv_op.pmap
    fwd = inv_op.forward
    est_n_dips = inv_op.est_n_dips
    vertno = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
    nv_tot = fwd['nsource']

    pmap_tot = np.zeros([len(pmaps), nv_tot])
    for it, bl in enumerate(pmaps):
        if est_n_dips[it] > 0:
            pmap_tot[it] = np.sum(bl, axis=0)

    tmin = 1
    stc = SourceEstimate(data=pmap_tot.T, vertices=vertno, tmin=tmin,
                         tstep=1, subject=subject)
    return stc


def _export_to_vol_stc(inv_op, subject=None):
    if not hasattr(inv_op, 'pmap'):
        raise AttributeError('Run the inversion algorithm first!!')

    pmaps = inv_op.pmap
    est_n_dips = inv_op.est_n_dips
    if len(inv_op.forward['src']) == 2:
        vertno = [inv_op.forward['src'][0]['vertno'],
                  inv_op.forward['src'][1]['vertno']]
    elif len(inv_op.forward['src']) == 1:
        vertno = inv_op.forward['src'][0]['vertno']
    else:
        raise ValueError
    nv_tot = inv_op.forward['nsource']
    pmap_tot = np.zeros([len(pmaps), nv_tot])

    for it, bl in enumerate(pmaps):
        if est_n_dips[it] > 0:
            pmap_tot[it] = np.sum(bl, axis=0)

    vol_stc = VolSourceEstimate(data=pmap_tot.T, vertices=vertno, tmin=1,
                                tstep=1, subject=subject)
    return vol_stc


def write_h5(fpath, inv_op, tmin=None, tmax=None, fmin=None,
             fmax=None, subsample=None, sbj=None, sbj_viz=None,
             data_path=None, fwd_path=None, cov_path=None,
             src_path=None, lf_path=None):
    _check_h5_installed()
    import h5py as h5

    f = h5.File(fpath, 'w')

    _pmap = f.create_group('prob_map')
    for i, _b in enumerate(inv_op.pmap):
        _pmap.create_dataset(str(i), data=_b)
    _est_cs = f.create_group('est_locs')
    for i, _e in enumerate(inv_op.est_locs):
        _est_cs.create_dataset(str(i), data=_e)
    _est_dip_moms = f.create_group('est_dip_moms')
    for i, _e in enumerate(inv_op.est_dip_moms):
        _est_dip_moms.create_dataset(str(i), data=_e)

    _est_ndips = f.create_dataset('est_n_dips',
                                  data=np.asarray(inv_op.est_n_dips))
    _model_sel = f.create_group('model_sel')
    for i, _m in enumerate(inv_op.model_sel):
        _model_sel.create_dataset(str(i), data=_m)
    _exponents = f.create_dataset('exponents',
                                  data=inv_op.e_pdf.exponents)
    if hasattr(inv_op, 'forward'):
        _ch_names = f.create_dataset('ch_names',
                                     shape=(len(inv_op.forward['info']['ch_names']),1),
                                     dtype='S10',
                                     data=list(ch.encode('ascii', 'ignore')
                                               for ch in inv_op.forward['info']['ch_names']))
    _lam = f.create_dataset('lambda', data=inv_op.lam)
    _sn = f.create_dataset('noise_std', data=inv_op.noise_std)
    _sq = f.create_dataset('dip_mom_std', data=inv_op.dip_mom_std)
    if inv_op.hyper_q:
        _fin_sq = f.create_dataset('final_dip_mom_std', data=inv_op.final_dip_mom_std)
        _est_dip_mom_std = f.create_group('est_dip_mom_std')
        for i, _sq in enumerate(inv_op.est_dip_mom_std):
            _est_dip_mom_std.create_dataset(str(i), data=_sq)
    _np = f.create_dataset('n_parts', data=inv_op.n_parts)
    _ndm = f.create_dataset('max_n_dips', data=inv_op.max_n_dips)
    if tmin is not None:
        _tmin = f.create_dataset('tmin', data=tmin)
    if tmax is not None:
        _tmax = f.create_dataset('tmax', data=tmax)
    if fmin is not None:
        _fmin = f.create_dataset('fmin', data=fmin)
    if fmax is not None:
        _fmax = f.create_dataset('fmax', data=fmax)
    if subsample is not None:
        _subsample = f.create_dataset('subsample', data=subsample)
    if sbj is not None:
        _sbj = f.create_dataset('subject', data=sbj)
    if sbj_viz is not None:
        _sbj_viz = f.create_dataset('subject_viz', data=sbj_viz)
    if data_path is not None:
        _dpath = f.create_dataset('data_path', data=data_path)
    if fwd_path is not None:
        _fwdpath = f.create_dataset('fwd_path', data=fwd_path)
    if cov_path is not None:
        _covpath = f.create_dataset('cov_path', data=fwd_path)
    if src_path is not None:
        _srcpath = f.create_dataset('src_path', data=src_path)
    if lf_path is not None:
        _lfpath = f.create_dataset('lf_path', data=lf_path)
    f.close()
    print('SESAME solution written in {}'.format(fpath))
    return


def write_pkl(fpath, inv_op, sbj=None, sbj_viz=None, data_path=None,
              fwd_path=None, cov_path=None, src_path=None, lf_path=None,
              save_all=False):
    _check_pickle_installed()
    import pickle as pkl

    if hasattr(inv_op, 'forward'):
        inv_op.ch_names = inv_op.forward['info']['ch_names']
    if sbj is not None:
        inv_op.sbj = sbj
    if sbj_viz is not None:
        inv_op.sbj_viz = sbj_viz
    if data_path is not None:
        inv_op.d_path = data_path
        if not save_all:
            del inv_op.r_data
    if fwd_path is not None:
        inv_op.fwd_path = fwd_path
        if not save_all:
            del inv_op.forward, inv_op.source_space, inv_op.lead_field
    if cov_path is not None:
        inv_op.cov_path = cov_path
    if src_path is not None:
        inv_op.src_path = src_path
        if not save_all:
            del inv_op.source_space
    if lf_path is not None:
        inv_op.lf_path =lf_path
        if not save_all:
            del inv_op.lead_field
    pkl.dump(inv_op, open(fpath, 'wb'))
    print('SESAME solution written in {}'.format(fpath))
    return


def read_h5(fpath):
    """Load SESAME result from an HDF5 file.

    Parameters
    ----------
    fpath : :py:class:`~str`
        Path and filename of the .h5 file containing the result.

    Returns
    -------
    res : :py:class:`~dict`
        A Python dictionary containing the result

    Notes
    -----
    The keys of the returned dictionary are the following:

    * ch_names : :py:class:`~list` | 'Not available.'
        The channel names
    * cov_path : :py:class:`~str` | 'Not available.'
        Path and filename of the file containing the noise covariance.
    * data_path : :py:class:`~str` | 'Not available.'
        Path and filename of the file containing the data.
    * est_locs : :py:class:`~list` of :py:class:`~numpy.ndarray` of :py:class:`~int` | 'Not available.'
        The source space grid points indices in which a source is estimated.
    * est_n_dips : :py:class:`~list` of :py:class:`~int` | 'Not available.'
        The estimated number of dipoles.
    * est_dip_moms : :py:class:`~numpy.ndarray` , shape (est_n_dips[-1], n_ist, 3) | 'Not available.'
        The moment time courses of the dipoles estimated in the last iteration of SESAME.
    * est_dip_mom_std : :py:class:`~list` of :py:class:`~numpy.ndarray`, shape (n_iterations, ) | 'Not available.'
        Estimated values of the parameter ``dip_mom_std``. Each array in the list corresponds to a single particle.
        This only applies if ``hyper_q=True`` has been selected when instantiating :py:class:`~sesameeg.Sesame`.
    * exponents : :py:class:`~numpy.ndarray` | 'Not available.'
        Array whose entries represent points in the space of artificial
        distributions. It is used to keep track of the path followed
        by SESAME.
    * final_dip_mom_std : :py:class:`~float` | 'Not available.'
        The weighted average of the last estimated value of ``dip_mom_std`` in each particle.
        This only applies if ``hyper_q=True`` has been selected when instantiating :py:class:`~sesameeg.Sesame`.
    * fmax : :py:class:`~float` | None
        The last frequency (in Hertz) of the frequency band in which data have been analyzed.
    * fmin : :py:class:`~float` | None
        The first frequency (in Hertz) of the frequency band in which data have been analyzed.
    * fwd_path : :py:class:`~str` | 'Not available.'
        Path and filename of the file containing the forward model.
    * lambda : :py:class:`~float`
        The parameter of the Poisson prior pdf on the number of dipoles.
    * lf_path : :py:class:`~str` | 'Not available.'
        Path and filename of the file containing the lead field.
    * max_n_dips : :py:class:`~int`
        The maximum number of dipoles allowed in each particle.
    * model_sel : :py:class:`~list` of :py:class:`~numpy.ndarray` of :py:class:`~float`
        The model selection, i.e. the posterior distribution on the number
        of dipoles.
    * prob_map : :py:class:`~list` of :py:class:`~numpy.ndarray` of :py:class:`~float`, shape (est_n_dips, n_verts)
        Posterior probability map
    * noise_std : :py:class:`~float`
        The standard deviation of the noise distribution.
    * dip_mom_std : :py:class:`~float`
        The standard deviation of the prior pdf on the dipole moment.
    * src_path : :py:class:`~str` | 'Not available.'
        Path and filename of the file containing the source space grid.
    * subject : :py:class:`~str` | 'Not available.'
        The subject name.
    * subject_viz : :py:class:`~str` | 'Not available.'
        The name of the subject's FreeSurfer folder.
    * tmax : :py:class:`~float` | None
        The last instant (in seconds) of the time window in which data have been analyzed.
    * tmin : :py:class:`~float` | None
        The first instant (in seconds) of the time window in which data have been analyzed.

    .. note::
       Depending on the value of the ``estimate_all`` parameter used in the call of the
       :py:meth:`~sesameeg.Sesame.apply_sesame` method, the list returned in the fields
       ``est_locs``, ``est_n_dips`` and ``prob_map`` may contain either
       the corresponding quantity estimated at each iteration or only those estimated at the last iteration.

    """
    _check_h5_installed()
    import h5py as h5

    f = h5.File(fpath, 'r')
    res = dict()

    if 'est_n_dips' in f.keys():
        res['est_n_dips'] = list(f['est_n_dips'][:])
    else:
        res['est_n_dips'] = 'Not available.'

    if 'exponents' in f.keys():
        res['exponents'] = f['exponents'][:]
    else:
        res['exponents'] = 'Not available.'

    if 'ch_names' in f.keys():
        _temp = list(f['ch_names'][:].flatten())
        res['ch_names'] = list(x.decode('utf-8', 'ignore') for x in _temp)
        del _temp
    else:
        res['ch_names'] = 'Not available.'

    for _k in ['prob_map', 'est_locs', 'model_sel', 'est_dip_mom_std']:
        if _k in f.keys():
            res[_k] = list(f[_k][_key][:] for _key in sorted(f[_k].keys(),
                                                             key=lambda x: int(x)))
        else:
            res[_k] = 'Not available.'

    for _k in ['final_dip_mom_std', 'tmin', 'tmax', 'fmin', 'fmax', 'subsample']:
        if _k in f.keys():
            res[_k] = f[_k][()]
        else:
            res[_k] = None

    for _k in ['lambda', 'noise_std', 'dip_mom_std', 'max_n_dips',
               'subject', 'subject_viz', 'data_path', 'fwd_path',
               'cov_path', 'src_path', 'lf_path']:
        if _k in f.keys():
            res[_k] = f[_k][()]
        else:
            res[_k] = 'Not available.'

    if 'est_dip_moms' in f.keys():
        est_dip_moms_temp = np.asarray(list(f['est_dip_moms'][_key][:] for _key in sorted(f['est_dip_moms'].keys(),
                                                                            key=lambda x: int(x))))
        est_dip_moms_aux = np.zeros((res['est_locs'][-1].shape[0], est_dip_moms_temp.shape[0], 3))
        for i in range(est_dip_moms_temp.shape[0]):
            _temp = est_dip_moms_temp[i, :].reshape(-1, 3)
            for j in range(res['est_locs'][-1].shape[0]):
                est_dip_moms_aux[j, i, :] += _temp[j]
        res['est_dip_moms'] = est_dip_moms_aux
    f.close()
    return res
