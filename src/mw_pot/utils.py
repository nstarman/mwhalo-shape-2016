# -*- coding: utf-8 -*-

# Docstring
"""**DOCSTRING**.

Attributes
----------
REFR0
REFV0

"""

# __all__ = ["REFR0", "REFV0"]


###############################################################################
# IMPORTS

# GENERAL

from typing import Sequence, Union

import numpy as np
from scipy import integrate

from galpy import potential
from galpy.potential import Potential
from galpy.util import bovy_conversion


# PROJECT-SPECIFIC

from . import data


###############################################################################
# PARAMETERS

REFR0: float = 8.0  # kpc
REFV0: float = 220.0  # km / s


PotentialType = Union[Potential, Sequence[Potential]]


###############################################################################
# CODE
###############################################################################


def _get_data() -> tuple:

    # Read the necessary data
    # First read the surface densities
    surfrs, kzs, kzerrs = data.readBovyRix13kzdata()

    # Then the terminal velocities
    dsinl = 0.125
    cl_glon, cl_vterm, cl_corr = data.readClemens(dsinl=dsinl)
    mc_glon, mc_vterm, mc_corr = data.readMcClureGriffiths07(
        dsinl=dsinl, bin=True
    )
    mc16_glon, mc16_vterm, mc16_corr = data.readMcClureGriffiths16(
        dsinl=dsinl, bin=True
    )

    termdata = (cl_glon, cl_vterm, cl_corr, mc_glon, mc_vterm, mc_corr)
    termdata_mc16 = (
        mc16_glon,
        mc16_vterm,
        mc16_corr,
        mc_glon,
        mc_vterm,
        mc_corr,
    )

    return (surfrs, kzs, kzerrs), termdata, termdata_mc16


# /def


def _get_data_and_make_funcargs(
    fitc, ro, vo, fitvoro, c, dblexp, addpal5, addgd1, mc16, addgas
) -> tuple:
    """Helper function for repeated action.

    Parameters
    ----------
    fitc, ro, vo, fitvoro, c, dblexp, addpal5, addgd1, mc16, addgas

    Returns
    -------
    (surfrs, kzs, kzerrs), termdata, termdata_mc16, funcargs

    """
    (surfrs, kzs, kzerrs), termdata, termdata_mc16 = _get_data()

    funcargs = [
        c,
        surfrs,
        kzs,
        kzerrs,
        termdata,
        7.0,
        fitc,
        fitvoro,
        dblexp,
        addpal5,
        addgd1,
        ro,
        vo,
        addgas,
    ]

    if mc16:  # replace termdata
        funcargs[4] = termdata_mc16

    return (surfrs, kzs, kzerrs), termdata, termdata_mc16, funcargs


# /def


# --------------------------------------------------------------------------


def mass60(pot: PotentialType, ro: float = REFR0, vo: float = REFV0) -> float:
    """The mass at 60 kpc in 10^11 msolar.

    Other Parameters
    ----------------
    ro: float
    vo: float

    """
    tR = 60.0 / ro
    # Average r^2 FR/G
    return (
        -integrate.quad(
            lambda x: tR ** 2.0
            * potential.evaluaterforces(
                pot, tR * x, tR * np.sqrt(1.0 - x ** 2.0), phi=0.0
            ),
            0.0,
            1.0,
        )[0]
        * bovy_conversion.mass_in_1010msol(vo, ro)
        / 10.0
    )


# /def


def bulge_dispersion(
    pot: PotentialType, ro: float = REFR0, vo: float = REFV0
) -> float:
    """The expected dispersion in Baade's window, in km/s.

    Other Parameters
    ----------------
    ro: float
    vo: float

    """
    bar, baz = 0.0175, 0.068
    return (
        np.sqrt(
            1.0
            / pot[0].dens(bar, baz)
            * integrate.quad(
                lambda x: -potential.evaluatezforces(pot, bar, x, phi=0.0)
                * pot[0].dens(bar, x),
                baz,
                np.inf,
            )[0]
        )
        * ro
    )


# /def


def visible_dens(
    pot: Sequence[Potential],
    ro: float = REFR0,
    vo: float = REFV0,
    r: float = 1.0,
) -> float:
    """The visible surface density at 8 kpc from the center.

    Parameters
    ----------
    pot: Potential
    ro : float
        default REFR0
    vo : float
        default REFV0
    r: float
        default 1.0

    Returns
    -------
    float

    """
    if len(pot) == 4:
        return (
            2.0
            * (
                integrate.quad(
                    (
                        lambda zz: potential.evaluateDensities(
                            pot[1], r, zz, phi=0.0
                        )
                    ),
                    0.0,
                    2.0,
                )[0]
                + integrate.quad(
                    (
                        lambda zz: potential.evaluateDensities(
                            pot[3], r, zz, phi=0.0
                        )
                    ),
                    0.0,
                    2.0,
                )[0]
            )
            * bovy_conversion.surfdens_in_msolpc2(vo, ro)
        )
    else:
        return (
            2.0
            * integrate.quad(
                (
                    lambda zz: potential.evaluateDensities(
                        pot[1], r, zz, phi=0.0
                    )
                ),
                0.0,
                2.0,
            )[0]
            * bovy_conversion.surfdens_in_msolpc2(vo, ro)
        )


# /def


###############################################################################
# END
