# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   :
# AUTHOR  :
# PROJECT :
#
# ----------------------------------------------------------------------------

# Docstring
"""**DOCSTRING**.

description

Routine Listings
----------------

"""

__author__ = ""
# __copyright__ = "Copyright 2019, "
# __credits__ = [""]
# __license__ = "MIT"
# __version__ = "0.0.0"
# __maintainer__ = ""
# __email__ = ""
# __status__ = "Production"

__all__ = [
    # parameters
    "_RAPAL5",
    "_DECPAL5",
    "_TPAL5",
    # functions
    "radec_to_pal5xieta",
    "timeout_handler",
]


###############################################################################
# IMPORTS

# GENERAL

import numpy as np

from galpy.util import bovy_coords

# CUSTOM

# PROJECT-SPECIFIC


###############################################################################
# PARAMETERS

# Coordinate transformation routines
_RAPAL5 = 229.018 / 180.0 * np.pi
_DECPAL5 = -0.124 / 180.0 * np.pi


_TPAL5 = np.dot(
    np.array(
        [
            [np.cos(_DECPAL5), 0.0, np.sin(_DECPAL5)],
            [0.0, 1.0, 0.0],
            [-np.sin(_DECPAL5), 0.0, np.cos(_DECPAL5)],
        ]
    ),
    np.array(
        [
            [np.cos(_RAPAL5), np.sin(_RAPAL5), 0.0],
            [-np.sin(_RAPAL5), np.cos(_RAPAL5), 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
)


###############################################################################
# CODE
###############################################################################


@bovy_coords.scalarDecorator
@bovy_coords.degreeDecorator([0, 1], [0, 1])
def radec_to_pal5xieta(ra: float, dec: float, degree: bool = False) -> np.ndarray:
    """Convert ra, dec to Pal 5 coordinate xi, eta.

    Parameters
    ----------
    ra: float
    dec: float
    degree: bool
        default False

    Returns
    -------
    xieta: ndarray
        [n, 2] array

    """
    XYZ = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])

    phiXYZ = np.dot(_TPAL5, XYZ)
    eta = np.arcsin(phiXYZ[2])
    xi = np.arctan2(phiXYZ[1], phiXYZ[0])

    return np.array([xi, eta]).T


# /def


###############################################################################
# END
