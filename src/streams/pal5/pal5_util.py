# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   : pal5_util
# PROJECT : Pal 5 update MW pot constraints
#
# ----------------------------------------------------------------------------

# Docstring
"""Palomar 5 Utility.

Routing Listings
----------------
radec_to_pal5xieta
width_trailing
vdisp_trailing
timeout_handler
predict_pal5obs
looks_funny
pal5_lnlike
setup_sdf
pal5_dpmguess
pal5_data_2016
pal5_data_2019
pal5_data_total

References
----------
https://github.com/jobovy/mwhalo-shape-2016

"""

__author__ = "Jo Bovy"
__copyright__ = "Copyright 2016, 2020, "
__maintainer__ = "Nathaniel Starkman"

__all__ = [
    # functions
    "width_trailing",
    "vdisp_trailing",
    "timeout_handler",
    "predict_pal5obs",
    "looks_funny",
    "pal5_lnlike",
    "setup_sdf",
    "pal5_dpmguess",
    "force_pal5",
]


###############################################################################
# IMPORTS

# GENERAL

import copy
import signal
import pickle
from typing import Optional, Union, Sequence, Tuple

import numpy as np
from tqdm import tqdm
from scipy import interpolate

# CUSTOM

from galpy.actionAngle import actionAngleIsochroneApprox, estimateBIsochrone
from galpy.actionAngle import actionAngleTorus
from galpy.orbit import Orbit
from galpy.df import streamdf
from galpy.util import bovy_conversion, bovy_coords
from galpy import potential
from galpy.potential import Potential

# PROJECT-SPECIFIC

from ... import mw_pot
from ...mw_pot.utils import REFR0, REFV0
from ...utils.exceptions import timeout_handler

from . import data, utils

from .data import pal5_data_2016, pal5_data_2019, pal5_data_total
from .utils import (
    _RAPAL5,
    _DECPAL5,
    _TPAL5,
    radec_to_pal5xieta,
)


###############################################################################
# PARAMETERS

_REFR0, _REFV0 = REFR0, REFV0

# typing
PotentialType = Union[Potential, Sequence[Potential]]


###############################################################################
# ALL

__all__ += data.__all__
__all__ += utils.__all__


###############################################################################
# CODE
###############################################################################


def force_pal5(
    pot: PotentialType, dpal5: float, ro: float = REFR0, vo: float = REFV0
) -> Tuple[float]:
    """Return the force at Pal5.

    Parameters
    ----------
    pot: Potential, list
    dpal5: float
    ro, vo: float

    Return
    ------
    force: tuple
        [fx, fy, fz]

    """
    from galpy import potential
    from galpy.util import bovy_coords

    # First compute the location based on the distance
    l5, b5 = bovy_coords.radec_to_lb(229.018, -0.124, degree=True)
    X5, Y5, Z5 = bovy_coords.lbd_to_XYZ(l5, b5, dpal5, degree=True)
    R5, p5, Z5 = bovy_coords.XYZ_to_galcencyl(X5, Y5, Z5, Xsun=ro, Zsun=0.025)

    args: list = [pot, R5 / ro, Z5 / ro]
    kws: dict = {"phi": p5, "use_physical": True, "ro": ro, "vo": vo}

    return (
        potential.evaluateRforces(*args, **kws),
        potential.evaluatezforces(*args, **kws),
        potential.evaluatephiforces(*args, **kws),
    )


# /def


# ----------------------------------------------------------------------------


def width_trailing(sdf) -> float:
    """Return the FWHM width in arcmin for the trailing tail.

    Parameters
    ----------
    sdf: streamdf

    Returns
    -------
    width: float
        the width of the stream

    """
    # Go out to RA=245 deg
    trackRADec_trailing = bovy_coords.lb_to_radec(
        sdf._interpolatedObsTrackLB[:, 0],
        sdf._interpolatedObsTrackLB[:, 1],
        degree=True,
    )
    cindx = range(len(trackRADec_trailing))[
        np.argmin(np.fabs(trackRADec_trailing[:, 0] - 245.0))
    ]

    ws = np.zeros(cindx)
    for ii, cc in enumerate(range(1, cindx + 1)):
        xy = [sdf._interpolatedObsTrackLB[cc, 0], None, None, None, None, None]
        ws[ii] = np.sqrt(sdf.gaussApprox(xy=xy, lb=True, cindx=cc)[1][0, 0])

    return 2.355 * 60.0 * np.mean(ws)


# /def


# ----------------------------------------------------------------------------


def vdisp_trailing(sdf) -> float:
    """Return the velocity dispersion in km/s for the trailing tail.

    Parameters
    ----------
    sdf: streamdf

    Returns
    -------
    vdisp: float
        the width of the stream

    """
    # Go out to RA=245 deg
    trackRADec_trailing = bovy_coords.lb_to_radec(
        sdf._interpolatedObsTrackLB[:, 0],
        sdf._interpolatedObsTrackLB[:, 1],
        degree=True,
    )
    cindx = range(len(trackRADec_trailing))[
        np.argmin(np.fabs(trackRADec_trailing[:, 0] - 245.0))
    ]

    ws = np.zeros(cindx)
    for ii, cc in enumerate(range(1, cindx + 1)):
        xy = [sdf._interpolatedObsTrackLB[cc, 0], None, None, None, None, None]
        ws[ii] = np.sqrt(sdf.gaussApprox(xy=xy, lb=True, cindx=cc)[1][2, 2])

    return np.mean(ws)


# /def


# ----------------------------------------------------------------------------


def predict_pal5obs(
    pot_params: list,
    c: float,
    b: float = 1.0,
    pa: float = 0.0,
    sigv: Union[float, list] = 0.2,
    td: float = 10.0,
    dist: float = 23.2,
    pmra: float = -2.296,
    pmdec: float = -2.257,
    vlos: float = -58.7,
    ro: float = REFR0,
    vo: float = REFV0,
    singlec: bool = False,
    interpcs: Optional[float] = None,
    interpk: Optional[float] = None,
    isob: Optional[float] = None,
    nTrackChunks: int = 8,
    multi: Optional = None,
    trailing_only: bool = False,
    useTM: bool = False,
    verbose: bool = True,
) -> Tuple:
    """Predict Pal 5 Observed.

    Function that generates the location and velocity of the Pal 5 stream,
    its width, and its length for a given potential and progenitor
    phase-space position

    Parameters
    ----------
    pot_params: array
        parameters of a potential model
        (see MWPotential2014Likelihood.setup_potential; only the basic
         parameters of the disk and halo are used,
         flattening is specified separately)
    c : float
        halo flattening
    b : float, optional
        (1.) halo-squashed
    pa: float, optional
        (0.) halo PA
    sigv : float, optional
        (0.4) velocity dispersion in km/s
        (can be array of same len as interpcs)
    td : float, optional
        (5.) stream age in Gyr (can be array of same len as interpcs)
    dist : float, optional
        (23.2) progenitor distance in kpc
    pmra : float, optional
        (-2.296) progenitor proper motion in RA * cos(Dec) in mas/yr
    pmdec : float, optional
        (-2.257) progenitor proper motion in Dec in mas/yr
    vlos : float, optional
        (-58.7) progenitor line-of-sight velocity in km/s
    ro : float, optional
        (project default) distance to the GC in kpc
    vo : float, optional
        (project default) circular velocity at R0 in km/s
    singlec : bool, optional
        (False) if True, just compute the observables for a single c
    interpcs : float or None, optional
        (None) values of c at which to compute the model for interpolation
    nTrackChunks : int, optional
        (8) nTrackChunks input to streamdf
    multi : float or None, optional
        (None) multi input to streamdf
    isob : float or None, optional
        (None) if given, b parameter of actionAngleIsochroneApprox
    trailing_only : bool, optional
        (False) if True, only predict the trailing arm
    useTM : bool, optional
        (True) use the TorusMapper to compute the track
    verbose : bool, optional
        (True) print messages


    Returns
    -------
    trackRADec_trailing_out : array
        trailing track in RA, Dec
        all arrays with the shape of c
    trackRADec_leading_out : array
        leading track in RA, Dec
        all arrays with the shape of c
    trackRAVlos_trailing_out : array
        trailing track in RA, Vlos
        all arrays with the shape of c
    trackRAVlos_leading_out : array
        leading track in RA, Vlos
        all arrays with the shape of c
    width_out : float
        trailing width in arcmin
    length_out : float
        trailing length in deg
    interpcs : array
    success : bool

    """
    from ...mw_pot import MWPotential2014Likelihood

    # First compute the model for all cs at which we will interpolate
    interpcs: list
    if singlec:
        interpcs = [c]
    elif interpcs is None:
        interpcs = [
            0.5,
            0.75,
            1.0,
            1.25,
            1.55,
            1.75,
            2.0,
            2.25,
            2.5,
            2.75,
            3.0,
        ]
    else:
        interpcs = copy.deepcopy(interpcs)  # bc we might want to remove some
    if isinstance(sigv, float):
        sigv: list = [sigv for i in interpcs]
    if isinstance(td, float):
        td: list = [td for i in interpcs]
    if isinstance(isob, float) or isob is None:
        isob: list = [isob for i in interpcs]

    prog: Orbit = Orbit(
        [229.018, -0.124, dist, pmra, pmdec, vlos],
        radec=True,
        ro=ro,
        vo=vo,
        solarmotion=[-11.1, 24.0, 7.25],
    )

    # Setup the model
    sdf_trailing_varyc: list = []
    sdf_leading_varyc: list = []
    ii: int = 0
    ninterpcs: int = len(interpcs)
    this_useTM: bool = copy.deepcopy(useTM)
    this_nTrackChunks: int = nTrackChunks
    ntries: int = 0

    while ii < ninterpcs:
        ic = interpcs[ii]
        pot = MWPotential2014Likelihood.setup_potential(
            pot_params, ic, False, False, ro, vo, b=b, pa=pa
        )
        success: bool = True
        # wentIn = ntries != 0
        # Make sure this doesn't run forever
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)
        try:
            tsdf_trailing, tsdf_leading = setup_sdf(
                pot,
                prog,
                sigv[ii],
                td[ii],
                ro,
                vo,
                multi=multi,
                isob=isob[ii],
                nTrackChunks=this_nTrackChunks,
                trailing_only=trailing_only,
                verbose=verbose,
                useTM=this_useTM,
            )
        except Exception:
            # Catches errors and time-outs
            success = False
        signal.alarm(0)
        # Check for calc. issues
        if not success or (
            not this_useTM and looks_funny(tsdf_trailing, tsdf_leading)
        ):
            # Try again with TM
            this_useTM = True
            this_nTrackChunks = 21  # might as well

            # wentIn= True
            # print("Here",ntries,success)
            # sys.stdout.flush()

            ntries += 1
        else:
            success = not this_useTM
            # if wentIn:
            #    print(success)
            #    sys.stdout.flush()
            ii += 1
            # reset
            this_useTM = useTM
            this_nTrackChunks = nTrackChunks
            ntries = 0
            # Add to the list
            sdf_trailing_varyc.append(tsdf_trailing)
            sdf_leading_varyc.append(tsdf_leading)

    if not singlec and len(sdf_trailing_varyc) <= 1:
        # Almost everything bad!!
        return (
            np.zeros((len(c), 1001, 2)),
            np.zeros((len(c), 1001, 2)),
            np.zeros((len(c), 1001, 2)),
            np.zeros((len(c), 1001, 2)),
            np.zeros((len(c))),
            np.zeros((len(c))),
            [],
            success,
        )
    # Compute the track properties for each model
    trackRADec_trailing: np.ndarray = np.zeros(
        (len(interpcs), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )
    trackRADec_leading: np.ndarray = np.zeros(
        (len(interpcs), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )
    trackRAVlos_trailing: np.ndarray = np.zeros(
        (len(interpcs), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )
    trackRAVlos_leading: np.ndarray = np.zeros(
        (len(interpcs), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )
    # TODO put in distances (can use _interpolatedObsTrackLB[:,2],)
    width: np.ndarray = np.zeros(len(interpcs))
    length: np.ndarray = np.zeros(len(interpcs))
    for ii in range(len(interpcs)):
        trackRADec_trailing[ii] = bovy_coords.lb_to_radec(
            sdf_trailing_varyc[ii]._interpolatedObsTrackLB[:, 0],
            sdf_trailing_varyc[ii]._interpolatedObsTrackLB[:, 1],
            degree=True,
        )
        if not trailing_only:
            trackRADec_leading[ii] = bovy_coords.lb_to_radec(
                sdf_leading_varyc[ii]._interpolatedObsTrackLB[:, 0],
                sdf_leading_varyc[ii]._interpolatedObsTrackLB[:, 1],
                degree=True,
            )
        trackRAVlos_trailing[ii][:, 0] = trackRADec_trailing[ii][:, 0]
        trackRAVlos_trailing[ii][:, 1] = sdf_trailing_varyc[
            ii
        ]._interpolatedObsTrackLB[:, 3]
        if not trailing_only:
            trackRAVlos_leading[ii][:, 0] = trackRADec_leading[ii][:, 0]
            trackRAVlos_leading[ii][:, 1] = sdf_leading_varyc[
                ii
            ]._interpolatedObsTrackLB[:, 3]
        width[ii] = width_trailing(sdf_trailing_varyc[ii])
        length[ii] = sdf_trailing_varyc[ii].length(
            ang=True, coord="customra", threshold=0.3
        )
    if singlec:
        # if wentIn:
        #    print(success)
        #    sys.stdout.flush()
        return (
            trackRADec_trailing,
            trackRADec_leading,
            trackRAVlos_trailing,
            trackRAVlos_leading,
            width,
            length,
            interpcs,
            success,
        )
    # Interpolate; output grids
    trackRADec_trailing_out: np.ndarray = np.zeros(
        (len(c), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )
    trackRADec_leading_out: np.ndarray = np.zeros(
        (len(c), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )
    trackRAVlos_trailing_out: np.ndarray = np.zeros(
        (len(c), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )
    trackRAVlos_leading_out: np.ndarray = np.zeros(
        (len(c), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )

    if interpk is None:
        interpk = np.amin([len(interpcs) - 1, 3])

    for ii in range(sdf_trailing_varyc[0].nInterpolatedTrackChunks):
        ip = interpolate.InterpolatedUnivariateSpline(
            interpcs, trackRADec_trailing[:, ii, 0], k=interpk, ext=0
        )
        trackRADec_trailing_out[:, ii, 0] = ip(c)
        ip = interpolate.InterpolatedUnivariateSpline(
            interpcs, trackRADec_trailing[:, ii, 1], k=interpk, ext=0
        )
        trackRADec_trailing_out[:, ii, 1] = ip(c)
        ip = interpolate.InterpolatedUnivariateSpline(
            interpcs, trackRAVlos_trailing[:, ii, 0], k=interpk, ext=0
        )
        trackRAVlos_trailing_out[:, ii, 0] = ip(c)
        ip = interpolate.InterpolatedUnivariateSpline(
            interpcs, trackRAVlos_trailing[:, ii, 1], k=interpk, ext=0
        )
        trackRAVlos_trailing_out[:, ii, 1] = ip(c)

        if not trailing_only:
            ip = interpolate.InterpolatedUnivariateSpline(
                interpcs, trackRADec_leading[:, ii, 0], k=interpk, ext=0
            )
            trackRADec_leading_out[:, ii, 0] = ip(c)
            ip = interpolate.InterpolatedUnivariateSpline(
                interpcs, trackRADec_leading[:, ii, 1], k=interpk, ext=0
            )
            trackRADec_leading_out[:, ii, 1] = ip(c)
            ip = interpolate.InterpolatedUnivariateSpline(
                interpcs, trackRAVlos_leading[:, ii, 0], k=interpk, ext=0
            )
            trackRAVlos_leading_out[:, ii, 0] = ip(c)
            ip = interpolate.InterpolatedUnivariateSpline(
                interpcs, trackRAVlos_leading[:, ii, 1], k=interpk, ext=0
            )
            trackRAVlos_leading_out[:, ii, 1] = ip(c)
    # /for

    ip = interpolate.InterpolatedUnivariateSpline(
        interpcs, width, k=interpk, ext=0
    )
    width_out = ip(c)
    ip = interpolate.InterpolatedUnivariateSpline(
        interpcs, length, k=interpk, ext=0
    )
    length_out = ip(c)

    return (
        trackRADec_trailing_out,
        trackRADec_leading_out,
        trackRAVlos_trailing_out,
        trackRAVlos_leading_out,
        width_out,
        length_out,
        interpcs,
        success,
    )


# /def


# ----------------------------------------------------------------------------


def looks_funny(tsdf_trailing, tsdf_leading) -> bool:
    """looks funny.

    Parameters
    ----------
    tsdf_trailing
    tsdf_leading

    Returns
    -------
    bool

    """
    radecs_trailing = bovy_coords.lb_to_radec(
        tsdf_trailing._interpolatedObsTrackLB[:, 0],
        tsdf_trailing._interpolatedObsTrackLB[:, 1],
        degree=True,
    )
    if not tsdf_leading is None:
        radecs_leading = bovy_coords.lb_to_radec(
            tsdf_leading._interpolatedObsTrackLB[:, 0],
            tsdf_leading._interpolatedObsTrackLB[:, 1],
            degree=True,
        )
    try:
        if radecs_trailing[0, 1] > 0.625:
            return True
        elif radecs_trailing[0, 1] < -0.1:
            return True
        elif np.any(
            (np.roll(radecs_trailing[:, 0], -1) - radecs_trailing[:, 0])[
                (radecs_trailing[:, 0] < 250.0)
                * (radecs_trailing[:, 1] > -1.0)
                * (radecs_trailing[:, 1] < 10.0)
            ]
            < 0.0
        ):
            return True
        elif not tsdf_leading is None and np.any(
            (np.roll(radecs_leading[:, 0], -1) - radecs_leading[:, 0])[
                (radecs_leading[:, 0] > 225.0)
                * (radecs_leading[:, 1] > -4.5)
                * (radecs_leading[:, 1] < 0.0)
            ]
            > 0.0
        ):
            return True
        elif False:  # np.isnan(width_trailing(tsdf_trailing)):
            return True
        elif np.isnan(
            tsdf_trailing.length(ang=True, coord="customra", threshold=0.3)
        ):
            return True
        elif (
            np.fabs(
                tsdf_trailing._dOdJpEig[0][2] / tsdf_trailing._dOdJpEig[0][1]
            )
            < 0.05
        ):
            return True
        else:
            return False
    except:
        return True


# /def


# ----------------------------------------------------------------------------


def pal5_lnlike(
    pos_radec,
    rvel_ra,
    trackRADec_trailing,
    trackRADec_leading,
    trackRAVlos_trailing,
    trackRAVlos_leading,
    width_out,
    length_out,
    interpcs,
    trailing_only=False,
) -> Sequence:  # last one so we can do *args
    """Pal 5 Ln-like.

    Returns array [nmodel,5] with log likelihood for each
    a) model,
    b) data set
        (trailing position,
         leading position,
         trailing vlos,
         actual width,
         actual length)

    Parameters
    ----------
    pos_radec
    rvel_ra
    trackRADec_trailing
    trackRADec_leading
    trackRAVlos_trailing
    trackRAVlos_leading
    width_out
    length_out
    interpcs
    trailing_only: bool, optional

    Returns
    -------
    out: (nmodel, 5) ndarray

    """
    nmodel = trackRADec_trailing.shape[0]
    out = np.zeros((nmodel, 5)) - 1e15
    for nn in range(nmodel):
        # Interpolate trailing RA, Dec track
        sindx = np.argsort(trackRADec_trailing[nn, :, 0])
        ipdec = interpolate.InterpolatedUnivariateSpline(
            trackRADec_trailing[nn, sindx, 0],
            trackRADec_trailing[nn, sindx, 1],
            k=1,
        )  # to be on the safe side
        tindx = pos_radec[:, 0] > 229.0
        out[nn, 0] = -0.5 * np.sum(
            (ipdec(pos_radec[tindx, 0]) - pos_radec[tindx, 1]) ** 2.0
            / pos_radec[tindx, 2] ** 2.0
        )
        # Interpolate leading RA,Dec track
        if not trailing_only:  # don't do trailing
            sindx = np.argsort(
                trackRADec_leading[nn, :, 0]
            )  # TODO throws error when 0!
            ipdec = interpolate.InterpolatedUnivariateSpline(
                trackRADec_leading[nn, sindx, 0],
                trackRADec_leading[nn, sindx, 1],
                k=1,
            )  # to be on the safe side
            tindx = pos_radec[:, 0] < 229.0
            out[nn, 1] = -0.5 * np.sum(
                (ipdec(pos_radec[tindx, 0]) - pos_radec[tindx, 1]) ** 2.0
                / pos_radec[tindx, 2] ** 2.0
            )
        # Interpolate trailing RA,Vlos track
        sindx = np.argsort(trackRAVlos_trailing[nn, :, 0])
        ipvlos = interpolate.InterpolatedUnivariateSpline(
            trackRAVlos_trailing[nn, sindx, 0],
            trackRAVlos_trailing[nn, sindx, 1],
            k=1,
        )  # to be on the safe side
        tindx = rvel_ra[:, 0] > 230.5
        out[nn, 2] = -0.5 * np.sum(
            (ipvlos(rvel_ra[tindx, 0]) - rvel_ra[tindx, 1]) ** 2.0
            / rvel_ra[tindx, 2] ** 2.0
        )
        out[nn, 3] = width_out[nn]
        out[nn, 4] = length_out[nn]

    out[np.isnan(out[:, 0]), 0] = -1e18
    out[np.isnan(out[:, 1]), 1] = -1e18
    out[np.isnan(out[:, 2]), 2] = -1e18

    return out


# /def

# ----------------------------------------------------------------------------


def setup_sdf(
    pot: potential.Potential,
    prog: Orbit,
    sigv: float,
    td: float,
    ro: float = REFR0,
    vo: float = REFV0,
    multi: Optional[bool] = None,
    nTrackChunks: float = 8,
    isob=None,
    trailing_only: bool = False,
    verbose: bool = True,
    useTM: bool = True,
):
    """Setup Stream Distribution Function.

    Parameters
    ----------
    pot : Potential
    prog : Orbit
        Progenitor
    sigv : float
    td : float
    ro : float
    vo : float
    multi
        default None
    nTrackChunks: float
        default 8
    isob
        default None
    trailing_only: bool
        default False
    verbose: bool
        default True
    useTM: bool
        default True

    Returns
    -------
    sdf_trailing, sdf_leading: ndarray or None

    """
    if isob is None:
        # Determine good one
        ts = np.linspace(0.0, 150.0, 1001)
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
        print(pot[2]._c, isob)
    if np.fabs(pot[2]._b - 1.0) > 0.05:
        aAI = actionAngleIsochroneApprox(
            pot=pot, b=isob, tintJ=1000.0, ntintJ=30000
        )
    else:
        aAI = actionAngleIsochroneApprox(pot=pot, b=isob)
    if useTM:
        aAT = actionAngleTorus(pot=pot, tol=0.001, dJ=0.0001)
    else:
        aAT = False

    trailing_kwargs = dict(
        progenitor=prog,
        pot=pot,
        aA=aAI,
        useTM=aAT,
        approxConstTrackFreq=True,
        leading=False,
        nTrackChunks=nTrackChunks,
        tdisrupt=td / bovy_conversion.time_in_Gyr(vo, ro),
        ro=ro,
        vo=vo,
        R0=ro,
        vsun=[-11.1, vo + 24.0, 7.25],
        custom_transform=_TPAL5,
        multi=multi,
    )

    try:
        sdf_trailing = streamdf(sigv / vo, **trailing_kwargs)
    except np.linalg.LinAlgError:
        sdf_trailing = streamdf(
            sigv / vo, nTrackIterations=0, **trailing_kwargs
        )

    if trailing_only:

        sdf_leading = None

    else:

        leading_kwargs = dict(
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
            custom_transform=_TPAL5,
            multi=multi,
        )

        try:
            sdf_leading = streamdf(sigv / vo, **leading_kwargs)
        except np.linalg.LinAlgError:
            sdf_leading = streamdf(
                sigv / vo, nTrackIterations=0, **leading_kwargs
            )

    return sdf_trailing, sdf_leading


# /def


# ----------------------------------------------------------------------------


def pal5_dpmguess(
    pot: PotentialType,
    doras=None,
    dodecs=None,
    dovloss=None,
    dmin=21.0,
    dmax=25.0,
    dstep=0.02,
    pmmin=-0.36,
    pmmax=0.36,
    pmstep=0.01,
    alongbfpm=False,
    ro=REFR0,
    vo=REFV0,
):
    """pal5_dpmguess.

    Parameters
    ----------
    pot: Potential
    doras : float or None, optional
        default None
    dodecs : float or None, optional
        default None
    dovloss : float or None, optional
        default None
    dmin=21 : float or None, optional
        default 0
    dmax=25 : float or None, optional
        default 0
    dstep=0 : float or None, optional
        default 02
    pmmin=-0 : float or None, optional
        default 36
    pmmax=0 : float or None, optional
        default 36
    pmstep=0 : float or None, optional
        default 01
    alongbfpm : float or None, optional
        default False
    ro : float or None, optional
        default REFR0
    vo : float or None, optional
        default REFV0

    Returns
    -------
    bestd
    bestpmoff
    lnl
    ds
    pmoffs

    """
    if doras is None:
        with open("pal5_stream_orbit_offset.pkl", "rb") as savefile:
            doras = pickle.load(savefile)
            dodecs = pickle.load(savefile)
            dovloss = pickle.load(savefile)

    ds = np.arange(dmin, dmax + dstep / 2.0, dstep)
    pmoffs = np.arange(pmmin, pmmax + pmstep / 2.0, pmstep)
    lnl = np.zeros((len(ds), len(pmoffs)))
    pos_radec, rvel_ra = pal5_data_total()

    print("Determining good distance and parallel proper motion...")

    for ii, d in tqdm(enumerate(ds)):
        for jj, pmoff in enumerate(pmoffs):
            if alongbfpm:
                pm = (d - 24.0) * 0.099 + 0.0769 + pmoff
            else:
                pm = pmoff
            progt = Orbit(
                [
                    229.11,
                    0.3,
                    d + 0.3,
                    -2.27 + pm,
                    -2.22 + pm * 2.257 / 2.296,
                    -58.5,
                ],
                radec=True,
                ro=ro,
                vo=vo,
                solarmotion=[-11.1, 24.0, 7.25],
            ).flip()
            ts = np.linspace(0.0, 3.0, 1001)
            progt.integrate(ts, pot)
            progt._orb.orbit[:, 1] *= -1.0
            progt._orb.orbit[:, 2] *= -1.0
            progt._orb.orbit[:, 4] *= -1.0
            toras, todecs, tovloss = (
                progt.ra(ts),
                progt.dec(ts),
                progt.vlos(ts),
            )
            # Interpolate onto common RA
            ipdec = interpolate.InterpolatedUnivariateSpline(toras, todecs)
            ipvlos = interpolate.InterpolatedUnivariateSpline(toras, tovloss)
            todecs = ipdec(doras) - dodecs
            tovloss = ipvlos(doras) - dovloss
            est_trackRADec_trailing = np.zeros((1, len(doras), 2))
            est_trackRADec_trailing[0, :, 0] = doras
            est_trackRADec_trailing[0, :, 1] = todecs
            est_trackRAVlos_trailing = np.zeros((1, len(doras), 2))
            est_trackRAVlos_trailing[0, :, 0] = doras
            est_trackRAVlos_trailing[0, :, 1] = tovloss
            lnl[ii, jj] = (
                np.sum(
                    pal5_lnlike(
                        pos_radec,
                        rvel_ra,
                        est_trackRADec_trailing,
                        est_trackRADec_trailing,  # hack
                        est_trackRAVlos_trailing,
                        est_trackRAVlos_trailing,  # hack
                        np.array([0.0]),
                        np.array([0.0]),
                        None,
                    )[0, :3:2]
                )
                - 0.5 * pm ** 2.0 / 0.186 ** 2.0
            )  # pm measurement

    bestd = ds[np.unravel_index(np.argmax(lnl), lnl.shape)[0]]

    if alongbfpm:
        bestpmoff = (
            (bestd - 24.0) * 0.099
            + 0.0769
            + pmoffs[np.unravel_index(np.argmax(lnl), lnl.shape)[1]]
        )
    else:
        bestpmoff = pmoffs[np.unravel_index(np.argmax(lnl), lnl.shape)[1]]

    return bestd, bestpmoff, lnl, ds, pmoffs


# /def


###############################################################################
# END
