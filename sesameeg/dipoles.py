# -*- coding: utf-8 -*-

# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#          Alberto Sorrentino <sorrentino@dima.unige.it>
#
# License: BSD (3-clause)

import numpy as np


class Dipole(object):
    """Single current dipole class for SESAME.

    Parameters
    ----------
    loc : int
        The dipole location (as an index of a brain grid).
    """
    def __init__(self, loc, zeta=None, phi=None, re_q=None, im_q=None):
        self.loc = loc
        if zeta:
            self.zeta = zeta
        if phi:
            self.phi = phi
        if re_q:
            self.re_q = re_q
        if im_q:
            self.im_q = im_q

    def __repr__(self):
        s = 'location : {0}'.format(self.loc)
        if self.zeta:
            s += ", z : %s" % str(np.float32(self.zeta))
        if self.phi:
            s += ", phi : %s" % str(np.float32(self.phi))
        if self.re_q:
            s += ", Re(q) : %s" % str(np.float32(self.re_q))
        if self.im_q:
            s += ", Im(q) : %s" % str(np.float32(self.im_q))

        return '<Dipole  |  {0}>'.format(s)
