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
pal5_data

References
----------
https://github.com/jobovy/mwhalo-shape-2016

"""

__author__ = "Jo Bovy"
__copyright__ = "Copyright 2016, 2020, "
__maintainer__ = "Nathaniel Starkman"

__all__ = [
    # parameters
    "_RAPAL5",
    "_DECPAL5",
    "_TPAL5",
    # functions
    "radec_to_pal5xieta",
    "width_trailing",
    "vdisp_trailing",
    "timeout_handler",
    "predict_pal5obs",
    "looks_funny",
    "pal5_lnlike",
    "setup_sdf",
    "pal5_dpmguess",
    "pal5_data",
]


###############################################################################
# IMPORTS

# GENERAL
import copy
import signal
import pickle
import numpy as np
import tqdm
from scipy import interpolate

from typing import Optional, Union

# CUSTOM
from galpy.actionAngle import actionAngleIsochroneApprox, estimateBIsochrone
from galpy.actionAngle import actionAngleTorus
from galpy.orbit import Orbit
from galpy.df import streamdf
from galpy.util import bovy_conversion, bovy_coords
from galpy import potential

# PROJECT-SPECIFIC
from . import MWPotential2014Likelihood
from .MWPotential2014Likelihood import _REFR0, _REFV0


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
def radec_to_pal5xieta(ra: float, dec: float, degree: bool = False):
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
    XYZ = np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )

    phiXYZ = np.dot(_TPAL5, XYZ)
    eta = np.arcsin(phiXYZ[2])
    xi = np.arctan2(phiXYZ[1], phiXYZ[0])

    return np.array([xi, eta]).T


# /def


# --------------------------------------------------------------------------


def width_trailing(sdf):
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

    width = 2.355 * 60.0 * np.mean(ws)
    return width


# /def


# --------------------------------------------------------------------------


def vdisp_trailing(sdf):
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

# --------------------------------------------------------------------------


def timeout_handler(signum, frame):
    """timeout_handler."""
    raise Exception("Calculation timed-out")


# /def

# --------------------------------------------------------------------------


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
    ro: float = _REFR0,
    vo: float = _REFV0,
    singlec: bool = False,
    interpcs: Optional[float] = None,
    interpk: Optional[float] = None,
    isob: Optional[float] = None,
    nTrackChunks: int = 8,
    multi: Optional = None,
    trailing_only: bool = False,
    useTM: bool = False,
    verbose: bool = True,
):
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
        sigv = [sigv for i in interpcs]
    if isinstance(td, float):
        td = [td for i in interpcs]
    if isinstance(isob, float) or isob is None:
        isob = [isob for i in interpcs]

    prog = Orbit(
        [229.018, -0.124, dist, pmra, pmdec, vlos],
        radec=True,
        ro=ro,
        vo=vo,
        solarmotion=[-11.1, 24.0, 7.25],
    )

    # Setup the model
    sdf_trailing_varyc = []
    sdf_leading_varyc = []
    ii: int = 0
    ninterpcs = len(interpcs)
    this_useTM = copy.deepcopy(useTM)
    this_nTrackChunks = nTrackChunks
    ntries = 0

    while ii < ninterpcs:
        ic = interpcs[ii]
        pot = MWPotential2014Likelihood.setup_potential(
            pot_params, ic, False, False, ro, vo, b=b, pa=pa
        )
        success = True
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
        except:
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
    trackRADec_trailing = np.zeros(
        (len(interpcs), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )
    trackRADec_leading = np.zeros(
        (len(interpcs), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )
    trackRAVlos_trailing = np.zeros(
        (len(interpcs), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )
    trackRAVlos_leading = np.zeros(
        (len(interpcs), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )
    # TODO put in distances (can use _interpolatedObsTrackLB[:,2],)
    width = np.zeros(len(interpcs))
    length = np.zeros(len(interpcs))
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
    trackRADec_trailing_out = np.zeros(
        (len(c), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )
    trackRADec_leading_out = np.zeros(
        (len(c), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )
    trackRAVlos_trailing_out = np.zeros(
        (len(c), sdf_trailing_varyc[0].nInterpolatedTrackChunks, 2)
    )
    trackRAVlos_leading_out = np.zeros(
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


# --------------------------------------------------------------------------


def looks_funny(tsdf_trailing, tsdf_leading):
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

# --------------------------------------------------------------------------


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
):  # last one so we can do *args
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
    

    """
    nmodel = trackRADec_trailing.shape[0]
    out = np.zeros((nmodel, 5)) - 1000000000000000.0
    for nn in range(nmodel):
        # Interpolate trailing RA,Dec track
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
        sindx = np.argsort(trackRADec_leading[nn, :, 0])
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
    out[np.isnan(out[:, 0]), 0] = -1000000000000000000.0
    out[np.isnan(out[:, 1]), 1] = -1000000000000000000.0
    out[np.isnan(out[:, 2]), 2] = -1000000000000000000.0
    return out


# /def

# --------------------------------------------------------------------------


def setup_sdf(
    pot: potential.Potential,
    prog,
    sigv,
    td,
    ro,
    vo,
    multi=None,
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
            sigv / vo,
            progenitor=prog,
            pot=pot,
            aA=aAI,
            useTM=aAT,
            approxConstTrackFreq=True,
            leading=False,
            nTrackChunks=nTrackChunks,
            nTrackIterations=0,
            tdisrupt=td / bovy_conversion.time_in_Gyr(vo, ro),
            ro=ro,
            vo=vo,
            R0=ro,
            vsun=[-11.1, vo + 24.0, 7.25],
            custom_transform=_TPAL5,
            multi=multi,
        )
    if trailing_only:
        return (sdf_trailing, None)
    try:
        sdf_leading = streamdf(
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
            custom_transform=_TPAL5,
            multi=multi,
        )
    except np.linalg.LinAlgError:
        sdf_leading = streamdf(
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
            custom_transform=_TPAL5,
            multi=multi,
        )
    return sdf_trailing, sdf_leading


# /def


# --------------------------------------------------------------------------


def pal5_dpmguess(
    pot,
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
    ro=_REFR0,
    vo=_REFV0,
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
        default _REFR0
    vo : float or None, optional
        default _REFV0

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
    pos_radec, rvel_ra = pal5_data()

    print("Determining good distance and parallel proper motion...")

    for ii, d in tqdm.tqdm(enumerate(ds)):
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


# --------------------------------------------------------------------------


def pal5_data():
    """Palomar 5 Data.

    Returns
    -------
    pos_radec: [n, 3] array
    rvel_ra: [n, 3] array

    """
    pos_radec = np.array(
        [
            [241.48, 6.41, 0.09],
            [240.98, 6.15, 0.09],
            [240.48, 6.20, 0.09],
            [239.98, 5.81, 0.09],
            [239.48, 5.64, 0.09],
            [238.48, 5.38, 0.09],
            [237.98, 5.14, 0.09],
            [233.61, 3.17, 0.06],
            [233.11, 2.88, 0.06],
            [232.61, 2.54, 0.06],
            [232.11, 2.23, 0.06],
            [231.61, 2.04, 0.06],
            [231.11, 1.56, 0.06],
            [230.11, 0.85, 0.06],
            [229.61, 0.54, 0.06],
            [228.48, -0.77, 0.11],
            [228.11, -1.16, 0.14],
            [227.73, -1.28, 0.11],
            [227.23, -2.03, 0.17],
            [226.55, -2.59, 0.14],
        ]
    )
    rvel_ra = np.array(
        [
            [225 + 15 * 15 / 60 + 48.19 * 0.25 / 60, -55.9, 1.2],
            [225 + 15 * 15 / 60 + 49.70 * 0.25 / 60, -56.9, 0.4],
            [225 + 15 * 15 / 60 + 52.60 * 0.25 / 60, -56.0, 0.6],
            [225 + 15 * 15 / 60 + 54.79 * 0.25 / 60, -57.6, 1.6],
            [225 + 15 * 15 / 60 + 56.11 * 0.25 / 60, -57.9, 0.7],
            [225 + 15 * 15 / 60 + 57.05 * 0.25 / 60, -55.6, 1.5],
            [225 + 15 * 15 / 60 + 58.26 * 0.25 / 60, -56.4, 1.0],
            [225 + 15 * 15 / 60 + 58.89 * 0.25 / 60, -55.9, 0.3],
            [225 + 15 * 15 / 60 + 59.52 * 0.25 / 60, -59.0, 0.4],
            [225 + 16 * 15 / 60 + 02.00 * 0.25 / 60, -58.0, 0.8],
            [225 + 16 * 15 / 60 + 03.61 * 0.25 / 60, -57.7, 2.5],
            [225 + 16 * 15 / 60 + 04.81 * 0.25 / 60, -57.2, 2.7],
            [225 + 16 * 15 / 60 + 06.54 * 0.25 / 60, -57.1, 0.2],
            [225 + 16 * 15 / 60 + 07.75 * 0.25 / 60, -60.6, 0.3],
            [225 + 16 * 15 / 60 + 08.51 * 0.25 / 60, -60.9, 3.3],
            [225 + 16 * 15 / 60 + 19.83 * 0.25 / 60, -56.9, 1.0],
            [225 + 16 * 15 / 60 + 23.11 * 0.25 / 60, -58.0, 2.5],
            [225 + 16 * 15 / 60 + 34.71 * 0.25 / 60, -58.2, 3.8],
            [225 + 16 * 15 / 60 + 08.66 * 0.25 / 60, -56.8, 0.7],
            [225 + 16 * 15 / 60 + 09.58 * 0.25 / 60, -57.7, 0.3],
            [225 + 15 * 15 / 60 + 52.84 * 0.25 / 60, -55.7, 0.6],
            [225 + 15 * 15 / 60 + 56.21 * 0.25 / 60, -55.9, 0.7],
            [225 + 16 * 15 / 60 + 05.26 * 0.25 / 60, -54.3, 0.3],
            [225 + 17 * 15 / 60 + 09.99 * 0.25 / 60, -57.0, 0.4],
            [225 + 17 * 15 / 60 + 34.55 * 0.25 / 60, -56.5, 3.1],
            [225 + 17 * 15 / 60 + 58.32 * 0.25 / 60, -57.5, 3.3],
            [225 + 18 * 15 / 60 + 04.96 * 0.25 / 60, -57.7, 2.6],
            [225 + 18 * 15 / 60 + 18.92 * 0.25 / 60, -57.6, 3.6],
            [225 + 18 * 15 / 60 + 35.89 * 0.25 / 60, -56.7, 1.3],
            [225 + 19 * 15 / 60 + 21.42 * 0.25 / 60, -61.7, 3.1],
            [225 + 21 * 15 / 60 + 51.16 * 0.25 / 60, -55.6, 0.4],
            [225 + 24 * 15 / 60 + 04.85 * 0.25 / 60, -56.5, 2.6],
            [225 + 24 * 15 / 60 + 13.00 * 0.25 / 60, -50.0, 2.4],
            [225 + 28 * 15 / 60 + 39.20 * 0.25 / 60, -56.6, 1.4],
            [225 + 28 * 15 / 60 + 49.34 * 0.25 / 60, -52.4, 3.8],
            [225 + 34 * 15 / 60 + 19.31 * 0.25 / 60, -55.8, 1.8],
            [225 + 34 * 15 / 60 + 31.90 * 0.25 / 60, -52.7, 4.0],
            [225 + 34 * 15 / 60 + 56.51 * 0.25 / 60, -51.9, 1.6],
            [225 + 45 * 15 / 60 + 10.57 * 0.25 / 60, -45.6, 2.6],
            [225 + 46 * 15 / 60 + 49.44 * 0.25 / 60, -48.0, 2.4],
            [225 + 48 * 15 / 60 + 57.99 * 0.25 / 60, -46.7, 2.3],
            [225 + 55 * 15 / 60 + 24.13 * 0.25 / 60, -41.0, 2.7],
            [240 + 0 * 15 / 60 + 45.41 * 0.25 / 60, -41.1, 2.8],
            [240 + 1 * 15 / 60 + 12.59 * 0.25 / 60, -40.8, 2.5],
            [240 + 3 * 15 / 60 + 29.59 * 0.25 / 60, -45.2, 3.9],
            [240 + 4 * 15 / 60 + 05.53 * 0.25 / 60, -44.9, 4.0],
            [240 + 4 * 15 / 60 + 33.28 * 0.25 / 60, -45.1, 3.5],
            [240 + 13 * 15 / 60 + 40.97 * 0.25 / 60, -41.1, 3.4],
            [240 + 16 * 15 / 60 + 44.79 * 0.25 / 60, -44.0, 3.0],
            [240 + 16 * 15 / 60 + 51.73 * 0.25 / 60, -43.5, 2.5],
            [225 + 8 * 15 / 60 + 07.15 * 0.25 / 60, -57.8, 1.1],
            [225 + 8 * 15 / 60 + 17.50 * 0.25 / 60, -62.0, 2.3],
            [225 + 10 * 15 / 60 + 39.02 * 0.25 / 60, -58.0, 1.0],
            [225 + 11 * 15 / 60 + 09.04 * 0.25 / 60, -66.9, 2.1],
            [225 + 11 * 15 / 60 + 21.70 * 0.25 / 60, -53.8, 1.1],
            [225 + 12 * 15 / 60 + 45.44 * 0.25 / 60, -52.5, 2.2],
            [225 + 13 * 15 / 60 + 40.44 * 0.25 / 60, -58.6, 1.4],
            [225 + 13 * 15 / 60 + 54.40 * 0.25 / 60, -59.8, 3.7],
            [225 + 14 * 15 / 60 + 09.32 * 0.25 / 60, -57.9, 3.5],
            [225 + 14 * 15 / 60 + 17.18 * 0.25 / 60, -59.2, 1.7],
            [225 + 14 * 15 / 60 + 20.71 * 0.25 / 60, -56.7, 2.3],
            [225 + 14 * 15 / 60 + 34.63 * 0.25 / 60, -59.1, 1.3],
            [225 + 15 * 15 / 60 + 16.47 * 0.25 / 60, -58.6, 2.3],
            [225 + 15 * 15 / 60 + 50.43 * 0.25 / 60, -55.7, 2.3],
            [225 + 16 * 15 / 60 + 01.54 * 0.25 / 60, -58.7, 1.4],
            [225 + 16 * 15 / 60 + 34.95 * 0.25 / 60, -59.7, 0.4],
            [225 + 16 * 15 / 60 + 56.20 * 0.25 / 60, -58.7, 0.2],
        ]
    )
    return pos_radec, rvel_ra


# /def

###############################################################################
# END
