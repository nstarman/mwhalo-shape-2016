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
plotRotcurve
plotKz
pltTerm
plotPot
plotDens
plot_samples
plotFit

"""

# __all__ = [
#     ""
# ]


###############################################################################
# IMPORTS

# GENERAL

from typing import Union, Sequence

import numpy as np

from galpy import potential
from galpy.potential import Potential
from galpy.util import bovy_plot, bovy_conversion

import matplotlib.pyplot as plt
import corner


# PROJECT-SPECIFIC

from .utils import REFR0, REFV0


###############################################################################
# PARAMETERS

PotentialType = Union[Potential, Sequence[Potential]]


###############################################################################
# CODE
###############################################################################


def plotRotcurve(pot: PotentialType) -> None:
    """Plot Terminal Velocity.

    Parameters
    ----------
    pot: potential

    """
    potential.plotRotcurve(
        pot, xrange=[0.0, 4.0], color="k", lw=2.0, yrange=[0.0, 1.4], gcf=True
    )
    # Constituents
    line1 = potential.plotRotcurve(pot[0], overplot=True, color="k", ls="-.", lw=2.0)
    line2 = potential.plotRotcurve(pot[1], overplot=True, color="k", ls="--", lw=2.0)
    line3 = potential.plotRotcurve(pot[2], overplot=True, color="k", ls=":", lw=2.0)
    # Add legend
    plt.legend(
        (line1[0], line2[0], line3[0]),
        (r"$\mathrm{Bulge}$", r"$\mathrm{Disk}$", r"$\mathrm{Halo}$"),
        loc="upper right",  # bbox_to_anchor=(.91,.375),
        numpoints=8,
        prop={"size": 16},
        frameon=False,
    )

    return None


# /def


# --------------------------------------------------------------------------


def plotKz(
    pot: PotentialType,
    surfrs: Sequence,
    kzs: Sequence,
    kzerrs: Sequence,
    ro: float = REFR0,
    vo: float = REFV0,
) -> float:
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
    plt.errorbar(
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
    plt.errorbar(
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


# --------------------------------------------------------------------------


def plotTerm(
    pot: PotentialType, termdata: Sequence, ro: float = REFR0, vo: float = REFV0,
) -> None:
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


# --------------------------------------------------------------------------


def plotPot(pot: PotentialType) -> None:
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


# --------------------------------------------------------------------------


def plotDens(pot: PotentialType):
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


# --------------------------------------------------------------------------


def plot_samples(
    samples,
    fitc,
    fitvoro,
    addpal5=False,
    addgd1=False,
    ro=REFR0,
    vo=REFV0,
    savefig=None,
):
    """Plot Samples.

    Parameters
    ----------
    samples : list
    fitc : bool
    fitvoro : bool
    addpal5 : bool, optional
    addgd1 : bool, optional
    ro : float, optional
    vo : float, optional

    Returns
    -------
    None

    """
    labels = [
        r"$f_d$",
        r"$f_h$",
        r"$h_R / \mathrm{kpc}$",
        r"$h_z / \mathrm{pc}$",
        r"$r_s / \mathrm{kpc}$",
    ]
    ranges = [(0.0, 1.0), (0.0, 1.0), (2.0, 4.49), (150.0, 445.0), (0.0, 39.0)]
    if fitvoro:
        labels.extend([r"$R_0 / \mathrm{kpc}$", r"$V_c(R_0) / \mathrm{km\,s}^{-1}$"])
        ranges.extend([(7.0, 8.7), (200.0, 240.0)])
    if fitc:
        labels.append(r"$c/a$")
        if addpal5 or addgd1:
            ranges.append((0.5, 1.5))
        else:
            ranges.append((0.0, 4.0))

    subset = np.ones(len(samples), dtype="bool")
    subset[5:7] = False
    plotsamples = samples[subset]
    plotsamples[2] = np.exp(samples[2]) * ro
    plotsamples[3] = np.exp(samples[3]) * 1000.0 * ro
    plotsamples[4] = np.exp(samples[4]) * ro

    if fitvoro:
        plotsamples[5] = samples[7] * ro
        plotsamples[6] = samples[8] * vo

    return_ = corner.corner(
        plotsamples.T,
        quantiles=[0.16, 0.5, 0.84],
        labels=labels,
        show_titles=True,
        title_args={"fontsize": 12},
        range=ranges,
    )

    if savefig is not None:
        plt.savefig(savefig)

    return return_


# /def


# --------------------------------------------------------------------------


def plotFit(
    pot, kzdata, termdata, ro=REFR0, vo=REFV0, suptitle=None, savefig=None,
):
    """Plot Fit to Potential.

    Parameters
    ----------
    pot : Potential or list of Potential
    kzdata : tuple
    termdata : tuple
    ro : float, optional
    vo : float, optional
    suptitle : str, optional
    savefig : str, optional

    Returns
    -------
    fig : Figure

    """
    fig = plt.Figure()
    plt.subplot(1, 3, 1)
    plotRotcurve(pot=pot)
    plt.subplot(1, 3, 2)
    plotKz(
        pot=pot,
        surfrs=kzdata.surfrs,
        kzs=kzdata.kzs,
        kzerrs=kzdata.kzerrs,
        ro=ro,
        vo=vo,
    )
    plt.subplot(1, 3, 3)
    plotTerm(pot=pot, termdata=termdata, ro=ro, vo=vo)

    plt.suptitle(suptitle)

    fig.tight_layout()

    if savefig is not None:
        plt.savefig(savefig)

    return fig


# /def


# --------------------------------------------------------------------------

###############################################################################
# END
