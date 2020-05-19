#! /usr/bin/env python
"""A template for mne-python compatible packages."""

import codecs
import os
import re

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('sesameeg', '_version.py')
with open(ver_file) as f_ver:
    version_file = f_ver.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        __version__ = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")


DISTNAME = 'sesameeg'
DESCRIPTION = 'Sequential Monte Carlo algorithm for multi dipolar source modeling in MEEG.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'SESAMEEG developers'
MAINTAINER_EMAIL = None
URL = 'https://github.com/pybees/sesameeg/'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/pybees/sesameeg/'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scipy', 'mne']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      # maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
