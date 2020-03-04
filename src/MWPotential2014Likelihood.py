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
force_pal5
force_gd1
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

import numpy as np
from scipy import integrate
from matplotlib import pyplot

import astropy.units as u

from galpy import potential
from galpy.util import bovy_plot, bovy_conversion, bovy_coords


# PROJECT-SPECIFIC
from .data import (  # import here for backward compatibility
    readBovyRix13kzdata,
    readClemens,
    readMcClureGriffiths07,
    readMcClureGriffiths16,
)


###############################################################################
# PARAMETERS


_REFR0 = 8.0  # kpc
_REFV0 = 220.0  # km / s


###############################################################################
# CODE
###############################################################################


def like_func(
    params: list,
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
    # Check ranges

    if params[0] < 0.0 or params[0] > 1.0:
        return np.finfo(np.dtype(np.float64)).max
    elif params[1] < 0.0 or params[1] > 1.0:
        return np.finfo(np.dtype(np.float64)).max
    elif (1.0 - params[0] - params[1]) < 0.0 or (
        1.0 - params[0] - params[1]
    ) > 1.0:
        return np.finfo(np.dtype(np.float64)).max
    elif params[2] < np.log(1.0 / _REFR0) or params[2] > np.log(8.0 / _REFR0):
        return np.finfo(np.dtype(np.float64)).max
    elif params[3] < np.log(0.05 / _REFR0) or params[3] > np.log(1.0 / _REFR0):
        return np.finfo(np.dtype(np.float64)).max
    elif fitvoro and (
        params[7] <= 150.0 / _REFV0 or params[7] > 290.0 / _REFV0
    ):
        return np.finfo(np.dtype(np.float64)).max
    elif fitvoro and (params[8] <= 7.0 / _REFR0 or params[8] > 9.4 / _REFR0):
        return np.finfo(np.dtype(np.float64)).max
    elif fitc and (
        params[7 + 2 * fitvoro] <= 0.0 or params[7 + 2 * fitvoro] > 4.0
    ):
        return np.finfo(np.dtype(np.float64)).max

    # --------------------------------------------------------------------

    if fitvoro:
        ro, vo = _REFR0 * params[8], _REFV0 * params[7]

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
        out += (
            0.5 * (np.sqrt(2.0 * fp5[0] / fp5[1]) - 0.94) ** 2.0 / 0.05 ** 2.0
        )
        out += (
            0.5
            * (0.94 ** 2.0 * (fp5[0] + 0.8) + 2.0 * (fp5[1] + 1.82) + 0.2)
            ** 2.0
            / 0.6 ** 2.0
        )

    # GD-1
    if addgd1:
        # q = 0.95 +/- 0.04 + add'l
        fg1 = force_gd1(pot, ro, vo)
        out += (
            0.5
            * (np.sqrt(6.675 / 12.5 * fg1[0] / fg1[1]) - 0.95) ** 2.0
            / 0.04 ** 2.0
        )
        out += (
            0.5
            * (
                0.95 ** 2.0 * (fg1[0] + 2.51)
                + 6.675 / 12.5 * (fg1[1] + 1.47)
                + 0.05
            )
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


def pdf_func(params, *args):
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

    """
    return -like_func(params, *args)


# /def


# --------------------------------------------------------------------------


def setup_potential(
    params,
    c,
    fitc,
    dblexp,
    ro,
    vo,
    fitvoro: bool = False,
    b: float = 1.0,
    pa: float = 0.0,
    addgas: bool = False,
):
    """Set up potential.

    Parameters
    ----------
    params
    c
    fitc
    dblexp
    ro
    vo
    fitvoro: bool, optional
        default False
    b: float, optional
        default 1.0
    pa: float, optional
        Position Angle
        default 0.0
    addgas: bool, optional
        default False

    """
    pot: potential.Potential = [
        potential.PowerSphericalPotentialwCutoff(
            normalize=1.0 - params[0] - params[1], alpha=1.8, rc=1.9 / ro
        )
    ]

    if dblexp:
        if addgas:
            # add 13 Msun/pc^2
            gp = potential.DoubleExponentialDiskPotential(
                amp=0.03333
                * u.Msun
                / u.pc ** 3
                * np.exp(ro / 2.0 / np.exp(params[2]) / _REFR0),
                hz=150.0 * u.pc,
                hr=2.0 * np.exp(params[2]) * _REFR0 / ro,
                ro=ro,
                vo=vo,
            )
            gp.turn_physical_off()
            gprf = gp.Rforce(1.0, 0.0)
            dpf = params[0] + gprf
            if dpf < 0.0:
                dpf = 0.0
            pot.append(
                potential.DoubleExponentialDiskPotential(
                    normalize=dpf,
                    hr=np.exp(params[2]) * _REFR0 / ro,
                    hz=np.exp(params[3]) * _REFR0 / ro,
                )
            )
        else:
            pot.append(
                potential.DoubleExponentialDiskPotential(
                    normalize=params[0],
                    hr=np.exp(params[2]) * _REFR0 / ro,
                    hz=np.exp(params[3]) * _REFR0 / ro,
                )
            )
    else:
        pot.append(
            potential.MiyamotoNagaiPotential(
                normalize=params[0],
                a=np.exp(params[2]) * _REFR0 / ro,
                b=np.exp(params[3]) * _REFR0 / ro,
            )
        )

    if fitc:
        pot.append(
            potential.TriaxialNFWPotential(
                normalize=params[1],
                a=np.exp(params[4]) * _REFR0 / ro,
                c=params[7 + 2 * fitvoro],
                b=b,
                pa=pa,
            )
        )
    else:
        pot.append(
            potential.TriaxialNFWPotential(
                normalize=params[1],
                a=np.exp(params[4]) * _REFR0 / ro,
                c=c,
                b=b,
                pa=pa,
            )
        )

    if addgas:
        pot.append(gp)  # make sure it's the last

    return pot


# /def


# --------------------------------------------------------------------------
# forces


def force_pal5(pot, dpal5, ro, vo):
    """Return the force at Pal5.

    Parameters
    ----------
    pot: Potential
    dpal5: float
    ro, vo: float

    Return
    ------
    force: tuple
        [fx, fy, fz]

    """
    # First compute the location based on the distance
    l5, b5 = bovy_coords.radec_to_lb(229.018, -0.124, degree=True)
    X5, Y5, Z5 = bovy_coords.lbd_to_XYZ(l5, b5, dpal5, degree=True)
    R5, p5, Z5 = bovy_coords.XYZ_to_galcencyl(X5, Y5, Z5, Xsun=ro, Zsun=0.025)

    return (
        potential.evaluateRforces(
            pot, R5 / ro, Z5 / ro, phi=p5, use_physical=True, ro=ro, vo=vo
        ),
        potential.evaluatezforces(
            pot, R5 / ro, Z5 / ro, phi=p5, use_physical=True, ro=ro, vo=vo
        ),
        potential.evaluatephiforces(
            pot, R5 / ro, Z5 / ro, phi=p5, use_physical=True, ro=ro, vo=vo
        ),
    )


# /def


def force_gd1(pot, ro, vo):
    """Return the force at GD-1.

    Parameters
    ----------
    pot: Potential
    ro, vo: float

    Return
    ------
    force: tuple
        [fx, fy, fz]

    """
    # Just use R=12.5 kpc, Z= 6.675 kpc, phi=0
    R1 = 12.5
    Z1 = 6.675
    p1 = 0.0

    return (
        potential.evaluateRforces(
            pot, R1 / ro, Z1 / ro, phi=p1, use_physical=True, ro=ro, vo=vo
        ),
        potential.evaluatezforces(
            pot, R1 / ro, Z1 / ro, phi=p1, use_physical=True, ro=ro, vo=vo
        ),
        potential.evaluatephiforces(
            pot, R1 / ro, Z1 / ro, phi=p1, use_physical=True, ro=ro, vo=vo
        ),
    )


# /def


# --------------------------------------------------------------------------


def mass60(pot, ro=_REFR0, vo=_REFV0):
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


def bulge_dispersion(pot, ro=_REFR0, vo=_REFV0):
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


def visible_dens(pot, ro=_REFR0, vo=_REFV0, r=1.0):
    """The visible surface density at 8 kpc from the center.

    Parameters
    ----------
    pot: Potential
    ro : float
        default _REFR0
    vo : float
        default _REFV0
    r: float
        default 1.0

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


def logprior_dlnvcdlnr(dlnvcdlnr: float):
    """Log Prior dlnvcdlnr.

    Parameters
    ----------
    dlnvcdlnr : float

    """
    sb = 0.04
    if dlnvcdlnr > sb or dlnvcdlnr < -0.5:
        return -np.finfo(np.dtype(np.float64)).max
    return np.log((sb - dlnvcdlnr) / sb) - (sb - dlnvcdlnr) / sb


# /def


###############################################################################
# PLOTS


def plotRotcurve(pot):
    """Plot Terminal Velocity.

    Parameters
    ----------
    pot: potential

    """
    potential.plotRotcurve(
        pot, xrange=[0.0, 4.0], color="k", lw=2.0, yrange=[0.0, 1.4], gcf=True
    )
    # Constituents
    line1 = potential.plotRotcurve(
        pot[0], overplot=True, color="k", ls="-.", lw=2.0
    )
    line2 = potential.plotRotcurve(
        pot[1], overplot=True, color="k", ls="--", lw=2.0
    )
    line3 = potential.plotRotcurve(
        pot[2], overplot=True, color="k", ls=":", lw=2.0
    )
    # Add legend
    pyplot.legend(
        (line1[0], line2[0], line3[0]),
        (r"$\mathrm{Bulge}$", r"$\mathrm{Disk}$", r"$\mathrm{Halo}$"),
        loc="upper right",  # bbox_to_anchor=(.91,.375),
        numpoints=8,
        prop={"size": 16},
        frameon=False,
    )

    return None


# /def


def plotKz(pot, surfrs, kzs, kzerrs, ro=_REFR0, vo=_REFV0):
    """Plot Terminal Velocity.

    Parameters
    ----------
    pot: potential
    surfrs: array-like
    kzs: array-like
    kzerrs: array-like

    Other Parameters
    ----------------
    ro: float
    vo: float

    """
    krs = np.linspace(4.0 / ro, 10.0 / ro, 1001)
    modelkz = np.array(
        [
            -potential.evaluatezforces(pot, kr, 1.1 / ro)
            * bovy_conversion.force_in_2piGmsolpc2(vo, ro)
            for kr in krs
        ]
    )
    bovy_plot.bovy_plot(
        krs * ro,
        modelkz,
        "-",
        color="0.6",
        lw=2.0,
        xlabel=r"$R\ (\mathrm{kpc})$",
        ylabel=r"$F_{Z}(R,|Z| = 1.1\,\mathrm{kpc})\ (2\pi G\,M_\odot\,\mathrm{pc}^{-2})$",
        semilogy=True,
        yrange=[10.0, 1000.0],
        xrange=[4.0, 10.0],
        zorder=0,
        gcf=True,
    )
    pyplot.errorbar(
        ro - 8.0 + surfrs,
        kzs,
        yerr=kzerrs,
        marker="o",
        elinewidth=1.0,
        capsize=3,
        zorder=1,
        color="k",
        linestyle="none",
    )
    pyplot.errorbar(
        [ro],
        [69.0],
        yerr=[6.0],
        marker="d",
        ms=10.0,
        elinewidth=1.0,
        capsize=3,
        zorder=10,
        color="0.4",
        linestyle="none",
    )
    # Do an exponential fit to the model Kz and return the scale length
    indx = krs < 9.0 / ro
    p = np.polyfit(krs[indx], np.log(modelkz[indx]), 1)
    return -1.0 / p[0]


# /def


def plotTerm(pot, termdata, ro=_REFR0, vo=_REFV0):
    """Plot Terminal Velocity.

    Parameters
    ----------
    pot: potential
    termdata: tuple
        cl_glon, cl_vterm, cl_corr, mc_glon, mc_vterm, mc_corr

    Other Parameters
    ----------------
    ro: float
    vo: float

    """
    mglons = np.linspace(-90.0, -20.0, 1001)
    pglons = np.linspace(20.0, 90.0, 1001)
    mterms = np.array([potential.vterm(pot, mgl) * vo for mgl in mglons])
    pterms = np.array([potential.vterm(pot, pgl) * vo for pgl in pglons])
    bovy_plot.bovy_plot(
        mglons,
        mterms,
        "-",
        color="0.6",
        lw=2.0,
        zorder=0,
        xlabel=r"$\mathrm{Galactic\ longitude\, (deg)}$",
        ylabel=r"$\mathrm{Terminal\ velocity}\, (\mathrm{km\,s}^{-1})$",
        xrange=[-100.0, 100.0],
        yrange=[-150.0, 150.0],
        gcf=True,
    )
    bovy_plot.bovy_plot(
        pglons, pterms, "-", color="0.6", lw=2.0, zorder=0, overplot=True
    )
    cl_glon, cl_vterm, cl_corr, mc_glon, mc_vterm, mc_corr = termdata
    bovy_plot.bovy_plot(cl_glon, cl_vterm, "ko", overplot=True)
    bovy_plot.bovy_plot(mc_glon - 360.0, mc_vterm, "ko", overplot=True)

    return None


# /def


def plotPot(pot):
    """Plot Potentials.

    Parameters
    ----------
    pot: potential

    """
    potential.plotPotentials(
        pot,
        rmin=0.0,
        rmax=1.5,
        nrs=201,
        zmin=-0.5,
        zmax=0.5,
        nzs=201,
        ncontours=21,
        justcontours=True,
        gcf=True,
    )
    return None


# /def


def plotDens(pot):
    """Plot Density.

    Parameters
    ----------
    pot: potential

    """
    potential.plotDensities(
        pot,
        rmin=0.01,
        rmax=1.5,
        nrs=201,
        zmin=-0.5,
        zmax=0.5,
        nzs=201,
        ncontours=21,
        log=True,
        justcontours=True,
        gcf=True,
    )
    return None


# /def


# -------------------------------------------------------------------------


# def readClemens(
#     dsinl=0.5 / 8.0, fpath="data/mwpot14data/clemens1985_table2.dat"
# ):
#     data = np.loadtxt(fpath, delimiter="|", comments="#",)
#     glon = data[:, 0]
#     vterm = data[:, 1]
#     # Remove l < 40 and l > 80
#     indx = (glon > 40.0) * (glon < 80.0)
#     glon = glon[indx]
#     vterm = vterm[indx]
#     if bin:
#         # Bin in l=1 bins
#         glon, vterm = binlbins(glon, vterm, dl=1.0)
#         # Remove nan, because 1 bin is empty
#         indx = ~np.isnan(glon)
#         glon = glon[indx]
#         vterm = vterm[indx]
#     # Calculate correlation matrix
#     singlon = np.sin(glon / 180.0 * np.pi)
#     corr = calc_corr(singlon, dsinl)
#     return (glon, vterm, np.linalg.inv(corr))


# # /def


# def readMcClureGriffiths(
#     dsinl=0.5 / 8.0,
#     bin=True,
#     fpath="data/mwpot14data/McClureGriffiths2007.dat",
# ):
#     data = np.loadtxt(fpath, comments="#")
#     glon = data[:, 0]
#     vterm = data[:, 1]
#     # Remove l > 320 and l > 80
#     indx = (glon < 320.0) * (glon > 280.0)
#     glon = glon[indx]
#     vterm = vterm[indx]
#     if bin:
#         # Bin in l=1 bins
#         glon, vterm = binlbins(glon, vterm, dl=1.0)
#     # Calculate correlation matrix
#     singlon = np.sin(glon / 180.0 * np.pi)
#     corr = calc_corr(singlon, dsinl)
#     return (glon, vterm, np.linalg.inv(corr))


# # /def


# def readMcClureGriffiths16(
#     dsinl=0.5 / 8.0,
#     bin=True,
#     fpath="data/mwpot14data/McClureGriffiths2016.dat",
# ):
#     data = np.loadtxt(fpath, comments="#", delimiter="&",)
#     glon = data[:, 0]
#     vterm = data[:, 1]
#     # Remove l < 30 and l > 80
#     indx = (glon > 40.0) * (glon < 80.0)
#     glon = glon[indx]
#     vterm = vterm[indx]
#     if bin:
#         # Bin in l=1 bins
#         glon, vterm = binlbins(glon, vterm, dl=1.0)
#     # Calculate correlation matrix
#     singlon = np.sin(glon / 180.0 * np.pi)
#     corr = calc_corr(singlon, dsinl)
#     return (glon, vterm, np.linalg.inv(corr))


# # /def

# # -------------------------------------------------------------------------


# def calc_corr(singlon, dsinl):
#     # Calculate correlation matrix
#     corr = np.zeros((len(singlon), len(singlon)))
#     for ii in range(len(singlon)):
#         for jj in range(len(singlon)):
#             corr[ii, jj] = np.exp(-np.fabs(singlon[ii] - singlon[jj]) / dsinl)
#     corr = 0.5 * (corr + corr.T)
#     return corr + 10.0 ** -10.0 * np.eye(len(singlon))  # for stability


# # /def


# def binlbins(glon, vterm, dl=1.0):
#     minglon, maxglon = (
#         np.floor(np.amin(glon)),
#         np.floor(np.amax(glon)),
#     )
#     minglon, maxglon = int(minglon), int(maxglon)
#     nout = maxglon - minglon + 1
#     glon_out = np.zeros(nout)
#     vterm_out = np.zeros(nout)
#     for ii in range(nout):
#         indx = (glon > minglon + ii) * (glon < minglon + ii + 1)
#         glon_out[ii] = np.mean(glon[indx])
#         vterm_out[ii] = np.mean(vterm[indx])
#     return (glon_out, vterm_out)


# # /def


##############################################################################
# END
