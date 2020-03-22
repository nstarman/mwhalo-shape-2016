# -*- coding: utf-8 -*-
# see LICENSE.rst

# ----------------------------------------------------------------------------
#
# TITLE   :
# AUTHOR  :
# PROJECT :
#
# ----------------------------------------------------------------------------

"""Milky Way Potential Subpackage.

Routine Listings
----------------
MWPotential2014Likelihood
utils
plot
REFR0
REFV0

"""

__author__ = "Nathaniel Starkman"
__copyright__ = "Copyright 2020, "
__credits__ = ["Jo Bovy"]

# __all__ = [
#     ""
# ]


##############################################################################
# IMPORTS

# PROJECT-SPECIFIC

from .utils import REFR0, REFV0
from .likelihood import like_func
from .potential import (
    fit,
    sample,
    sample_multi,
    setup_potential,
)
from .plot import plot_samples


##############################################################################
# PARAMETERS

# _REFR0: float = REFR0
# _REFV0: float = REFV0


##############################################################################
# END
