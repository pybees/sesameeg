# -*- coding: utf-8 -*-
"""The check functions."""
# Authors: Gianvittorio Luria <luria@dima.unige.it>
#
# License: BSD (3-clause)


def _check_h5_installed(strict=True):
    """Aux function."""
    try:
        import h5py
        return h5py
    except ImportError:
        if strict is True:
            raise RuntimeError('For this functionality to work, the h5py '
                               'library is required.')
        else:
            return False
