# -*- coding: utf-8 -*-

# Authors: Gianvittorio Luria <luria@dima.unige.it>
#          Sara Sommariva <sommariva@dima.unige.it>
#          Alberto Sorrentino <sorrentino@dima.unige.it>
#
# License: BSD (3-clause)


class Dipole(object):
    """Single current dipole class for SESAME.

    Parameters
    ----------
    loc : :py:class:`int`
        The dipole location (as an index of a brain grid).
    """
    def __init__(self, loc):
        self.loc = loc

    def __repr__(self):
        s = 'location : {0}'.format(self.loc)

        return '<Dipole  |  {0}>'.format(s)
