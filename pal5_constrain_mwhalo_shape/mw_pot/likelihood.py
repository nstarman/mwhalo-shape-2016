# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   : MWPotential2014Likelihood
# PROJECT : Pal 5 update MW pot constraints
#
# ----------------------------------------------------------------------------

# Docstring
"""Milky Way Potential (2014 version) Likelihood.

Routing Listings
----------------
like_func
pdf_func
setup_potential
mass60
bulge_dispersion
visible_dens
logprior_dlnvcdlnr
plotRotcurve
plotKz
plotTerm
plotPot
plotDens
readClemens
readMcClureGriffiths
readMcClureGriffiths16
calc_corr
binlbins

References
----------
https://github.com/jobovy/mwhalo-shape-2016

"""

__author__ = "Jo Bovy"
__copyright__ = "Copyright 2016, 2020, "
__maintainer__ = "Nathaniel Starkman"

# __all__ = [
#     ""
# ]


###############################################################################
# IMPORTS

# GENERAL

import functools
from typing import Sequence, Tuple, Union, Optional

import numpy as np
from scipy import optimize
import emcee
from tqdm import tqdm

import astropy.units as u

from galpy import potential
from galpy.potential import Potential
from galpy.util import bovy_conversion

import matplotlib.pyplot as plt


# CUSTOM

import bovy_mcmc  # TODO not need


# PROJECT-SPECIFIC

from .utils import (
    REFR0,
    REFV0,
    _get_data_and_make_funcargs,
    mass60,
    bulge_dispersion,
    visible_dens,
)

from . import plot


###############################################################################
# PARAMETERS

PotentialType = Union[Potential, Sequence[Potential]]


###############################################################################
# CODE
###############################################################################


def like_func(
    params: Sequence,
    c: float,
    surfrs: list,
    kzs,
    kzerrs,
    termdata,
    termsigma,
    fitc,
    fitvoro,
    dblexp,
    addpal5,
    addgd1,
    ro: float,
    vo: float,
    addgas: bool,
):
    """Likelihood Function.

    Parameters
    ----------
    params: list
    c: float
    surfrs: list
    kzs
    kzerrs
    termdata
    termsigma
    fitc
    fitvoro
    dblexp
    addpal5
    addgd1
    ro: float
    vo: float
    addgas: bool

    Returns
    -------
    float

    """
    # --------------------------------------------------------------------

    from .potential import setup_potential  # TODO how do not internally?

    # TODO use a more generic switcher so can use any stream
    from ..streams.pal5.pal5_util import force_pal5
    from ..streams.gd1.gd1_util import force_gd1

    # --------------------------------------------------------------------
    # Check ranges

    if params[0] < 0.0 or params[0] > 1.0:
        return np.finfo(np.dtype(np.float64)).max
    elif params[1] < 0.0 or params[1] > 1.0:
        return np.finfo(np.dtype(np.float64)).max
    elif (1.0 - params[0] - params[1]) < 0.0 or (1.0 - params[0] - params[1]) > 1.0:
        return np.finfo(np.dtype(np.float64)).max
    elif params[2] < np.log(1.0 / REFR0) or params[2] > np.log(8.0 / REFR0):
        return np.finfo(np.dtype(np.float64)).max
    elif params[3] < np.log(0.05 / REFR0) or params[3] > np.log(1.0 / REFR0):
        return np.finfo(np.dtype(np.float64)).max
    elif fitvoro and (params[7] <= 150.0 / REFV0 or params[7] > 290.0 / REFV0):
        return np.finfo(np.dtype(np.float64)).max
    elif fitvoro and (params[8] <= 7.0 / REFR0 or params[8] > 9.4 / REFR0):
        return np.finfo(np.dtype(np.float64)).max
    elif fitc and (params[7 + 2 * fitvoro] <= 0.0 or params[7 + 2 * fitvoro] > 4.0):
        return np.finfo(np.dtype(np.float64)).max

    # --------------------------------------------------------------------

    if fitvoro:
        ro, vo = REFR0 * params[8], REFV0 * params[7]

    # Setup potential
    pot = setup_potential(
        params, c, fitc, dblexp, ro, vo, fitvoro=fitvoro, addgas=addgas
    )

    # Calculate model surface density at surfrs
    modelkzs = np.empty_like(surfrs)
    for ii in range(len(surfrs)):
        modelkzs[ii] = -potential.evaluatezforces(
            pot, (ro - 8.0 + surfrs[ii]) / ro, 1.1 / ro, phi=0.0
        ) * bovy_conversion.force_in_2piGmsolpc2(vo, ro)
    out = 0.5 * np.sum((kzs - modelkzs) ** 2.0 / kzerrs ** 2.0)

    # Add terminal velocities
    vrsun = params[5]
    vtsun = params[6]
    cl_glon, cl_vterm, cl_corr, mc_glon, mc_vterm, mc_corr = termdata

    # Calculate terminal velocities at data glon
    cl_vterm_model = np.zeros_like(cl_vterm)
    for ii in range(len(cl_glon)):
        cl_vterm_model[ii] = potential.vterm(pot, cl_glon[ii])
    cl_vterm_model += vrsun * np.cos(cl_glon / 180.0 * np.pi) - vtsun * np.sin(
        cl_glon / 180.0 * np.pi
    )
    mc_vterm_model = np.zeros_like(mc_vterm)
    for ii in range(len(mc_glon)):
        mc_vterm_model[ii] = potential.vterm(pot, mc_glon[ii])
    mc_vterm_model += vrsun * np.cos(mc_glon / 180.0 * np.pi) - vtsun * np.sin(
        mc_glon / 180.0 * np.pi
    )
    cl_dvterm = (cl_vterm - cl_vterm_model * vo) / termsigma
    mc_dvterm = (mc_vterm - mc_vterm_model * vo) / termsigma
    out += 0.5 * np.sum(cl_dvterm * np.dot(cl_corr, cl_dvterm))
    out += 0.5 * np.sum(mc_dvterm * np.dot(mc_corr, mc_dvterm))

    # Rotation curve constraint
    out -= logprior_dlnvcdlnr(potential.dvcircdR(pot, 1.0, phi=0.0))

    # K dwarfs, Kz
    out += (
        0.5
        * (
            -potential.evaluatezforces(pot, 1.0, 1.1 / ro, phi=0.0)
            * bovy_conversion.force_in_2piGmsolpc2(vo, ro)
            - 67.0
        )
        ** 2.0
        / 36.0
    )
    # K dwarfs, visible
    out += 0.5 * (visible_dens(pot, ro, vo) - 55.0) ** 2.0 / 25.0
    # Local density prior
    localdens = potential.evaluateDensities(
        pot, 1.0, 0.0, phi=0.0
    ) * bovy_conversion.dens_in_msolpc3(vo, ro)
    out += 0.5 * (localdens - 0.102) ** 2.0 / 0.01 ** 2.0
    # Bulge velocity dispersion
    out += 0.5 * (bulge_dispersion(pot, ro, vo) - 117.0) ** 2.0 / 225.0
    # Mass at 60 kpc
    out += 0.5 * (mass60(pot, ro, vo) - 4.0) ** 2.0 / 0.7 ** 2.0

    # Pal5
    if addpal5:
        # q = 0.94 +/- 0.05 + add'l
        fp5 = force_pal5(pot, 23.46, ro, vo)
        out += 0.5 * (np.sqrt(2.0 * fp5[0] / fp5[1]) - 0.94) ** 2.0 / 0.05 ** 2.0
        out += (
            0.5
            * (0.94 ** 2.0 * (fp5[0] + 0.8) + 2.0 * (fp5[1] + 1.82) + 0.2) ** 2.0
            / 0.6 ** 2.0
        )

    # GD-1
    if addgd1:
        # q = 0.95 +/- 0.04 + add'l
        fg1 = force_gd1(pot, ro, vo)
        out += (
            0.5 * (np.sqrt(6.675 / 12.5 * fg1[0] / fg1[1]) - 0.95) ** 2.0 / 0.04 ** 2.0
        )
        out += (
            0.5
            * (0.95 ** 2.0 * (fg1[0] + 2.51) + 6.675 / 12.5 * (fg1[1] + 1.47) + 0.05)
            ** 2.0
            / 0.3 ** 2.0
        )

    # vc and ro measurements: vc=218 +/- 10 km/s, ro= 8.1 +/- 0.1 kpc
    out += (vo - 218.0) ** 2.0 / 200.0 + (ro - 8.1) ** 2.0 / 0.02

    if np.isnan(out):
        return np.finfo(np.dtype(np.float64)).max
    else:
        return out


# /def


# --------------------------------------------------------------------------


def pdf_func(params: Sequence, *args) -> float:
    """PDF function.

    the negative likelihood

    Parameters
    ----------
    params: list
    c: float
    surfrs: list
    kzs
    kzerrs
    termdata
    termsigma
    fitc
    fitvoro
    dblexp
    addpal5
    addgd1
    ro: float
    vo: float
    addgas: bool

    Returns
    -------
    float

    """
    return -like_func(params, *args)


# /def


# --------------------------------------------------------------------------


def logprior_dlnvcdlnr(dlnvcdlnr: float) -> float:
    """Log Prior dlnvcdlnr.

    Parameters
    ----------
    dlnvcdlnr : float

    Returns
    -------
    float

    """
    sb = 0.04
    if dlnvcdlnr > sb or dlnvcdlnr < -0.5:
        return -np.finfo(np.dtype(np.float64)).max
    return np.log((sb - dlnvcdlnr) / sb) - (sb - dlnvcdlnr) / sb


# /def


##############################################################################
# END
