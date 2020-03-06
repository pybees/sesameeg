import numpy as np
from ..utils import _check_h5_installed, _check_pickle_installed


def write_h5(fpath, inv_op, tmin=None, tmax=None, subsample=None,
             sbj=None, sbj_viz=None, data_path=None, fwd_path=None,
             src_path=None, lf_path=None):
    _check_h5_installed()
    import h5py as h5

    f = h5.File(fpath, 'w')

    _blob = f.create_group('prob_map')
    for i, _b in enumerate(inv_op.blob):
        _blob.create_dataset(str(i), data=_b)
    _est_cs = f.create_group('est_locs')
    for i, _e in enumerate(inv_op.est_locs):
        _est_cs.create_dataset(str(i), data=_e)
    if hasattr(inv_op, 's_q'):
        _est_q = f.create_group('est_q')
        for i, _e in enumerate(inv_op.est_q):
            _est_q.create_dataset(str(i), data=_e)
    elif hasattr(inv_op, 'q_in'):
        _est_im_q = f.create_group('est_im_q')
        for i, _e in enumerate(inv_op.est_im_q):
            _est_im_q.create_dataset(str(i), data=_e)
        _est_re_q = f.create_group('est_re_q')
        for i, _e in enumerate(inv_op.est_re_q):
            _est_re_q.create_dataset(str(i), data=_e)

    _est_ndips = f.create_dataset('est_n_dips',
                                  data=np.asarray(inv_op.est_n_dips))
    _model_sel = f.create_group('model_sel')
    for i, _m in enumerate(inv_op.model_sel):
        _model_sel.create_dataset(str(i), data=_m)
    _exponents = f.create_dataset('exponents',
                                  data=inv_op.emp.exponents)
    if hasattr(inv_op, 'forward'):
        _ch_names = f.create_dataset('ch_names',
                                     shape=(len(inv_op.forward['info']['ch_names']),1),
                                     dtype='S10',
                                     data=list(ch.encode('ascii', 'ignore')
                                               for ch in inv_op.forward['info']['ch_names']))
    _lam = f.create_dataset('lambda', data=inv_op.lam)
    _sn = f.create_dataset('sigma_noise', data=inv_op.s_noise)
    _sq = f.create_dataset('sigma_q', data=inv_op.s_q)
    if inv_op.hyper_q:
        _fin_sq = f.create_dataset('final_s_q', data=inv_op.final_s_q)
        _est_s_q = f.create_group('est_s_q')
        for i, _sq in enumerate(inv_op.est_s_q):
            _est_s_q.create_dataset(str(i), data=_sq)
    _np = f.create_dataset('n_parts', data=inv_op.n_parts)
    _ndm = f.create_dataset('n_max_dip', data=inv_op.N_dip_max)
    if tmin is not None:
        _tmin = f.create_dataset('tmin', data=tmin)
    if tmax is not None:
        _tmax = f.create_dataset('tmax', data=tmax)
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
    if src_path is not None:
        _srcpath = f.create_dataset('src_path', data=src_path)
    if lf_path is not None:
        _lfpath = f.create_dataset('lf_path', data=lf_path)
    f.close()
    print('SESAME solution written in {}'.format(fpath))
    return


def write_pkl(fpath, inv_op, tmin=None, tmax=None, subsample=None,
              sbj=None, sbj_viz=None, data_path=None, fwd_path=None,
              src_path=None, lf_path=None):
    _check_pickle_installed()
    import pickle as pkl

    if hasattr(inv_op, 'forward'):
        inv_op.ch_names = inv_op.forward['info']['ch_names']
    if tmin is not None:
        inv_op.t_min = tmin
    if tmax is not None:
        inv_op.t_max = tmax
    if subsample is not None:
        inv_op.subsample = subsample
    if sbj is not None:
        inv_op.sbj = sbj
    if sbj_viz is not None:
        inv_op.sbj_viz = sbj_viz
    if data_path is not None:
        inv_op.d_path = data_path
        del inv_op.i_data, inv_op.r_data
    if fwd_path is not None:
        inv_op.fwd_path = fwd_path
        del inv_op.forward, inv_op.source_space, inv_op.lead_field
    if src_path is not None:
        inv_op.src_path = src_path
        del inv_op.source_space
    if lf_path is not None:
        inv_op.lf_path =lf_path
        del inv_op.lead_field
    pkl.dump(inv_op, open(fpath, 'wb'))
    print('SESAME solution written in {}'.format(fpath))
    return


def read_h5(fpath):
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

    for _k in ['prob_map', 'est_locs', 'model_sel', 'est_s_q']:
        if _k in f.keys():
            res[_k] = list(f[_k][_key][:] for _key in sorted(f[_k].keys(),
                                                             key=lambda x: int(x)))
        else:
            res[_k] = 'Not available.'

    for _k in ['sigma_q', 'final_s_q', 'tmin', 'tmax', 'subsample']:
        if _k in f.keys():
            res[_k] = f[_k][()]

    for _k in ['lambda', 'sigma_noise', 'n_max_dip',
               'subject', 'subject_viz', 'data_path', 'fwd_path',
               'src_path', 'lf_path']:
        if _k in f.keys():
            res[_k] = f[_k][()]
        else:
            res[_k] = 'Not available.'

    if 'est_q' in f.keys():
        est_q_temp = np.asarray(list(f['est_q'][_key][:] for _key in sorted(f['est_q'].keys(),
                                                                            key=lambda x: int(x))))
        est_q_aux = np.zeros((res['est_locs'][-1].shape[0], est_q_temp.shape[0], 3))
        for i in range(est_q_temp.shape[0]):
            _temp = est_q_temp[i, :].reshape(-1, 3)
            for j in range(res['est_locs'][-1].shape[0]):
                est_q_aux[j, i, :] += _temp[j]
        res['est_q'] = est_q_aux
    f.close()
    return res
