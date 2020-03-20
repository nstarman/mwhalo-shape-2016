# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   : gd1_util
# PROJECT : Pal 5 update MW pot constraints
#
# ----------------------------------------------------------------------------

# Docstring
"""GD1 5 Utility.

Routing Listings
----------------

References
----------
https://github.com/jobovy/mwhalo-shape-2016

"""

__author__ = "Jo Bovy"
__copyright__ = "Copyright 2016, 2020, "
__maintainer__ = "Nathaniel Starkman"

# __all__ = [""]


###############################################################################
# IMPORTS

# GENERAL
import copy
from typing import Optional, Union, Sequence, Any

import numpy as np
from scipy import interpolate

from galpy import potential
from galpy.potential import Potential
from galpy.actionAngle import (
    actionAngleIsochroneApprox,
    estimateBIsochrone,
    actionAngleTorus,
)
from galpy.orbit import Orbit
from galpy.df import streamdf
from galpy.util import bovy_conversion


# PROJECT-SPECIFIC

from ...mw_pot import MWPotential2014Likelihood, REFR0, REFV0

from .data import gd1_data_2016, gd1_data_total
from .utils import phi12_to_lb_6d, convert_track_lb_to_phi12, _TKOP, force_gd1


###############################################################################
# PARAMETERS


###############################################################################
# CODE
###############################################################################


def predict_gd1obs(
    pot_params: Sequence[float],
    c: float,
    b: float = 1.0,
    pa: float = 0.0,
    sigv: float = 0.4,
    td: float = 10.0,
    dist: float = 10.2,
    pmphi1: float = -8.5,
    pmphi2: float = -2.05,
    phi1: float = 0.0,
    phi2: float = -1.0,
    vlos: float = -285.0,
    ro: float = REFR0,
    vo: float = REFV0,
    isob: Optional[bool] = None,
    nTrackChunks: int = 8,
    multi: Optional[Any] = None,
    useTM: bool = False,
    logpot: bool = False,
    verbose: bool = True,
):
    """Predict GD1-observed

    Function that generates the location and velocity of the GD-1 stream,
    its width, and its length for a given potential
    and progenitor phase-space position.

    Parameters
    ----------
    pot_params
        array with the parameters of a potential model
        (see MWPotential2014Likelihood.setup_potential;
         only the basic parameters of the disk and halo are used,
         flattening is specified separately)
    c
        halo flattening
    b
        (1.) halo-squashed
    pa
        (0.) halo PA
    sigv
        (0.365) velocity dispersion in km/s
    td
        (10.) stream age in Gyr
    phi1
        (0.) Progenitor's phi1 position
    phi2
        (-1.) Progenitor's phi2 position
    dist
        (10.2) progenitor distance in kpc
    pmphi1
        (-8.5) progenitor proper motion in phi1 in mas/yr
    pmphi2
        (-2.) progenitor proper motion in phi2 in mas/yr
    vlos
        (-250.) progenitor line-of-sight velocity in km/s
    ro
        (project default) distance to the GC in kpc
    vo
        (project default) circular velocity at R0 in km/s
    nTrackChunks
        (8) nTrackChunks input to streamdf
    multi
        (None) multi input to streamdf
    logpot
        (False) if True, use a logarithmic potential instead
    isob
        (None) if given, b parameter of actionAngleIsochroneApprox
    useTM
        (True) use the TorusMapper to compute the track
    verbose
        (True) print messages

    Notes
    ------
    2016-08-12 - Written - Bovy (UofT)

    """
    # Convert progenitor's phase-space position to l,b,...
    prog = Orbit(
        phi12_to_lb_6d(phi1, phi2, dist, pmphi1, pmphi2, vlos),
        lb=True,
        ro=ro,
        vo=vo,
        solarmotion=[-11.1, 24.0, 7.25],
    )
    if logpot:
        pot = potential.LogarithmicHaloPotential(normalize=1.0, q=c)
    else:
        pot = MWPotential2014Likelihood.setup_potential(
            pot_params, c, False, False, ro, vo, b=b, pa=pa
        )
    success: bool = True
    this_useTM = useTM

    try:
        sdf = setup_sdf(
            pot,
            prog,
            sigv,
            td,
            ro,
            vo,
            multi=multi,
            isob=isob,
            nTrackChunks=nTrackChunks,
            verbose=verbose,
            useTM=useTM,
            logpot=logpot,
        )
    except:
        # Catches errors and time-outs
        success: bool = False

    # Check for calc. issues
    if not success:
        return (np.zeros((1001, 6)) - 1000000.0, False)
        # Try again with TM
        this_useTM = True
        this_nTrackChunks = 21  # might as well
        sdf = setup_sdf(
            pot,
            prog,
            sigv,
            td,
            ro,
            vo,
            multi=multi,
            isob=isob,
            nTrackChunks=this_nTrackChunks,
            verbose=verbose,
            useTM=this_useTM,
            logpot=logpot,
        )
    else:
        success: bool = not this_useTM

    # Compute the track and convert it to phi12
    track_phi = convert_track_lb_to_phi12(sdf._interpolatedObsTrackLB)

    return track_phi, success


# /def


# ---------------------------------------------------------------------


def gd1_lnlike(
    posdata: Sequence,
    distdata: Sequence,
    pmdata: Sequence,
    rvdata: Sequence,
    track_phi: bool,
):
    """GD1 Log-Likelihood.

    Returns array [5] with log likelihood for each
    b) data set (phi2,dist,pmphi1,pmphi2,vlos)

    """
    out = np.zeros(5) - 1e15
    if np.any(np.isnan(track_phi)):
        return out
    # Interpolate all tracks to evaluate the likelihood
    sindx = np.argsort(track_phi[:, 0])
    # phi2
    ipphi2 = interpolate.InterpolatedUnivariateSpline(
        track_phi[sindx, 0], track_phi[sindx, 1], k=1
    )  # to be on the safe side
    out[0] = -0.5 * np.sum(
        (ipphi2(posdata[:, 0]) - posdata[:, 1]) ** 2.0 / posdata[:, 2] ** 2.0
    )
    # dist
    ipdist = interpolate.InterpolatedUnivariateSpline(
        track_phi[sindx, 0], track_phi[sindx, 2], k=1
    )  # to be on the safe side
    out[1] = -0.5 * np.sum(
        (ipdist(distdata[:, 0]) - distdata[:, 1]) ** 2.0 / distdata[:, 2] ** 2.0
    )
    # pmphi1
    ippmphi1 = interpolate.InterpolatedUnivariateSpline(
        track_phi[sindx, 0], track_phi[sindx, 4], k=1
    )  # to be on the safe side
    out[2] = -0.5 * np.sum(
        (ippmphi1(pmdata[:, 0]) - pmdata[:, 1]) ** 2.0 / pmdata[:, 3] ** 2.0
    )
    # pmphi2
    ippmphi2 = interpolate.InterpolatedUnivariateSpline(
        track_phi[sindx, 0], track_phi[sindx, 5], k=1
    )  # to be on the safe side
    out[3] = -0.5 * np.sum(
        (ippmphi2(pmdata[:, 0]) - pmdata[:, 2]) ** 2.0 / pmdata[:, 3] ** 2.0
    )
    # vlos
    ipvlos = interpolate.InterpolatedUnivariateSpline(
        track_phi[sindx, 0], track_phi[sindx, 3], k=1
    )  # to be on the safe side
    out[4] = -0.5 * np.sum(
        (ipvlos(rvdata[:, 0]) - rvdata[:, 2]) ** 2.0 / rvdata[:, 3] ** 2.0
    )

    return out


# /def


# ---------------------------------------------------------------------


def setup_sdf(
    pot: Sequence[Potential],
    prog: Orbit,
    sigv: float,
    td: float,
    ro: float = REFR0,
    vo: float = REFV0,
    multi: Optional[Any] = None,
    nTrackChunks: int = 8,
    isob: Optional[bool] = None,
    trailing_only: bool = False,
    verbose: bool = True,
    useTM: bool = True,
    logpot: bool = False,
):
    """Simple function to setup the stream model."""
    if isob is None:
        if True or logpot:  # FIXME, "if True"
            isob = 0.75
    if isob is False:  # FIXME, was "if False"
        # Determine good one
        ts = np.linspace(0.0, 15.0, 1001)
        # Hack!
        epot = copy.deepcopy(pot)
        epot[2]._b = 1.0
        epot[2]._b2 = 1.0
        epot[2]._isNonAxi = False
        epot[2]._aligned = True
        prog.integrate(ts, pot)
        estb = estimateBIsochrone(
            epot,
            prog.R(ts, use_physical=False),
            prog.z(ts, use_physical=False),
            phi=prog.phi(ts, use_physical=False),
        )
        if estb[1] < 0.3:
            isob = 0.3
        elif estb[1] > 1.5:
            isob = 1.5
        else:
            isob = estb[1]
        if verbose:
            print(pot[2]._c, isob, estb)

    if not logpot and np.fabs(pot[2]._b - 1.0) > 0.05:
        aAI = actionAngleIsochroneApprox(pot=pot, b=isob, tintJ=1000.0, ntintJ=30000)
    else:
        ts = np.linspace(0.0, 100.0, 10000)
        aAI = actionAngleIsochroneApprox(
            pot=pot, b=isob, tintJ=100.0, ntintJ=10000, dt=ts[1] - ts[0]
        )

    if useTM:
        aAT = actionAngleTorus(pot=pot, tol=0.001, dJ=0.0001)
    else:
        aAT = False

    try:
        sdf = streamdf(
            sigv / vo,
            progenitor=prog,
            pot=pot,
            aA=aAI,
            useTM=aAT,
            approxConstTrackFreq=True,
            leading=True,
            nTrackChunks=nTrackChunks,
            tdisrupt=td / bovy_conversion.time_in_Gyr(vo, ro),
            ro=ro,
            vo=vo,
            R0=ro,
            vsun=[-11.1, vo + 24.0, 7.25],
            custom_transform=_TKOP,
            multi=multi,
            nospreadsetup=True,
        )
    except np.linalg.LinAlgError:
        sdf = streamdf(
            sigv / vo,
            progenitor=prog,
            pot=pot,
            aA=aAI,
            useTM=aAT,
            approxConstTrackFreq=True,
            leading=True,
            nTrackChunks=nTrackChunks,
            nTrackIterations=0,
            tdisrupt=td / bovy_conversion.time_in_Gyr(vo, ro),
            ro=ro,
            vo=vo,
            R0=ro,
            vsun=[-11.1, vo + 24.0, 7.25],
            custom_transform=_TKOP,
            multi=multi,
        )

    return sdf


# /def


###############################################################################
# END
