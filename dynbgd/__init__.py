# Licensed under a 3-clause BSD style license - see LICENSE.rst

import ska_helpers

__version__ = ska_helpers.get_version(__package__)

from .clean import clean_file, clean_img


def test(*args, **kwargs):
    """
    Run py.test unit tests.
    """
    import testr
    return testr.test(*args, **kwargs)
