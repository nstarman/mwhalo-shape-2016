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

Routing Listings
----------------

"""

__author__ = "Jo Bovy"
__maintainer__ = "Nathaniel Starkman"

__all__ = [
    "fiducial_model",
    "model_vary_c_along_the_best_fit_line",
    "model-vary_b_for_c_1_pa_0",
    "model_vary_b_for_c_1_pa_45",
    "model_vary_d",
    "model_vary_pm_pll",
    "model_vary_pm_perp",
    "model_stream_struct_vary_c",
    "model_stream_struct_vary_vcirc",
    "model_stream_struct_vary_d",
    "model_stream_struct_vary_pm",
    "plot_all",
    "make_parser",
    "main",
]


###############################################################################
# IMPORTS

# GENERAL
import os
import os.path
import argparse
import pickle
import warnings
from typing import Optional


import numpy as np

from tqdm import tqdm

# galpy
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.orbit import Orbit
from galpy.df import streamdf
from galpy.util import bovy_plot, bovy_conversion, bovy_coords, save_pickles

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from matplotlib.ticker import NullFormatter
from matplotlib.cbook import MatplotlibDeprecationWarning

# PROJECT-SPECIFIC
# fmt: off
import sys; sys.path.insert(0, '../../../')
# fmt: on
from pal5_constrain_mwhalo_shape import mw_pot
from pal5_constrain_mwhalo_shape.mw_pot import MWPotential2014Likelihood
from pal5_constrain_mwhalo_shape.streams.pal5 import pal5_util


###############################################################################
# PARAMETERS

_MW_POT_SCRIPT_FOLDER = "../../0-create_MW_potential_2014/2019/"

save_figures = False

cmap = cm.plasma

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

REFR0 = mw_pot.REFR0
REFV0 = mw_pot.REFV0

pos_radec, rvel_ra = pal5_util.pal5_data_total()


###############################################################################
# CODE
###############################################################################


def plot_data_add_labels(
    radeconly=False,
    rvonly=False,
    color=sns.color_palette()[2],
    p1=(1, 2, 1),
    p2=(1, 2, 2),
    noxlabel_dec=False,
    noxlabel_vlos=False,
    noylabel=False,
):
    """plot_data_add_labels."""
    if noxlabel_dec or noylabel:
        nullfmt = NullFormatter()
    if not radeconly and not rvonly:
        plt.subplot(*p1)
    if not rvonly:
        if not noxlabel_dec:
            plt.xlabel(r"$\mathrm{RA}\,(\mathrm{degree})$")
        else:
            plt.gca().xaxis.set_major_formatter(nullfmt)
        if not noylabel:
            plt.ylabel(r"$\mathrm{Dec}\,(\mathrm{degree})$")
        else:
            plt.gca().yaxis.set_major_formatter(nullfmt)
        plt.xlim(250.0, 210.0)
        plt.ylim(-15.0, 9.0)
        bovy_plot._add_ticks()
        plt.errorbar(
            pos_radec[:, 0],
            pos_radec[:, 1],
            yerr=pos_radec[:, 2],
            ls="none",
            marker="o",
            color=color,
        )
    if radeconly:
        return None
    if not rvonly:
        plt.subplot(*p2)
    if not noxlabel_vlos:
        plt.xlabel(r"$\mathrm{RA}\,(\mathrm{degree})$")
    else:
        plt.gca().xaxis.set_major_formatter(nullfmt)
    if not noylabel:
        plt.ylabel(r"$V_{\mathrm{los}}\,(\mathrm{km\,s}^{-1})$")
    else:
        plt.gca().yaxis.set_major_formatter(nullfmt)
    plt.xlim(250.0, 221.0)
    plt.ylim(-80.0, 0.0)
    bovy_plot._add_ticks()
    plt.errorbar(
        rvel_ra[rvel_ra[:, 0] > 230.5, 0],
        rvel_ra[rvel_ra[:, 0] > 230.5, 1],
        yerr=rvel_ra[rvel_ra[:, 0] > 230.5, 2],
        ls="none",
        marker="o",
        color=color,
    )

    return None


# /def


def add_colorbar(vmin, vmax, clabel, save_figures=False):
    """Add Colorbar."""
    fig = plt.gcf()
    if save_figures:
        cbar_ax = fig.add_axes([0.9, 0.135, 0.025, 0.815])
    else:
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.975, 0.13, 0.025, 0.78])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, use_gridspec=True, format=r"$%.1f$")
    cbar.set_label(clabel)

    return None


# /def


def fiducial_model(
    sdf_trailing="output/sdf_trailing.pkl",
    sdf_leading="output/sdf_leading.pkl",
    threshold=0.3,
    ro=REFR0,
    vo=REFV0,
):
    """Fiducial Model.

    The fiducial model assumes a spherical halo, with the best-fit
    parameters from fitting to the MWPotential2014 data

    """
    p_b15 = [
        0.60122692,
        0.36273147,
        -0.97591502,
        -3.34169377,
        0.71877924,
        -0.01519337,
        -0.01928001,
    ]

    pot = MWPotential2014Likelihood.setup_potential(p_b15, 1.0, False, False, ro, vo)

    prog = Orbit(
        [229.018, -0.124, 23.2, -2.296, -2.257, -58.7],
        radec=True,
        ro=ro,
        vo=vo,
        solarmotion=[-11.1, 24.0, 7.25],
    )
    aAI = actionAngleIsochroneApprox(pot=pot, b=0.8)
    sigv = 0.2

    # ----------------------------------------------------------

    try:
        with open(sdf_trailing, "rb") as file:
            sdf_trailing = pickle.load(file)
    except Exception:
        sdf_trailing = streamdf(
            sigv / vo,
            progenitor=prog,
            pot=pot,
            aA=aAI,
            leading=False,
            nTrackChunks=11,
            tdisrupt=10.0 / bovy_conversion.time_in_Gyr(vo, ro),
            ro=ro,
            vo=vo,
            R0=ro,
            vsun=[-11.1, vo + 24.0, 7.25],
            custom_transform=pal5_util._TPAL5,
        )
        with open(sdf_trailing, "wb") as file:
            pickle.dump(sdf_trailing, file)

    try:
        with open(sdf_leading, "rb") as file:
            sdf_leading = pickle.load(file)
    except Exception:
        sdf_leading = streamdf(
            sigv / vo,
            progenitor=prog,
            pot=pot,
            aA=aAI,
            leading=True,
            nTrackChunks=11,
            tdisrupt=10.0 / bovy_conversion.time_in_Gyr(vo, ro),
            ro=ro,
            vo=vo,
            R0=ro,
            vsun=[-11.1, vo + 24.0, 7.25],
            custom_transform=pal5_util._TPAL5,
        )
        with open(sdf_leading, "wb") as file:
            pickle.dump(sdf_leading, file)

    # ----------------------------------------------------------

    print(
        "Angular length: %f deg (leading,trailing)=(%f,%f) deg"
        % (
            sdf_leading.length(ang=True, coord="customra", threshold=threshold)
            + sdf_trailing.length(ang=True, coord="customra", threshold=threshold),
            sdf_leading.length(ang=True, coord="customra", threshold=threshold),
            sdf_trailing.length(ang=True, coord="customra", threshold=threshold),
        )
    )
    print("Angular width (FWHM): %f arcmin" % (pal5_util.width_trailing(sdf_trailing)))
    print("Velocity dispersion: %f km/s" % (pal5_util.vdisp_trailing(sdf_trailing)))

    # ----------------------------------------------------------

    trackRADec_trailing = bovy_coords.lb_to_radec(
        sdf_trailing._interpolatedObsTrackLB[:, 0],
        sdf_trailing._interpolatedObsTrackLB[:, 1],
        degree=True,
    )
    trackRADec_leading = bovy_coords.lb_to_radec(
        sdf_leading._interpolatedObsTrackLB[:, 0],
        sdf_leading._interpolatedObsTrackLB[:, 1],
        degree=True,
    )
    lb_sample_trailing = sdf_trailing.sample(n=10000, lb=True)
    lb_sample_leading = sdf_leading.sample(n=10000, lb=True)
    radec_sample_trailing = bovy_coords.lb_to_radec(
        lb_sample_trailing[0], lb_sample_trailing[1], degree=True
    )
    radec_sample_leading = bovy_coords.lb_to_radec(
        lb_sample_leading[0], lb_sample_leading[1], degree=True
    )

    # ----------------------------------------------------------
    # plotting

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    bovy_plot.bovy_plot(
        trackRADec_trailing[:, 0],
        trackRADec_trailing[:, 1],
        color=sns.color_palette()[0],
        xrange=[250.0, 210.0],
        yrange=[-15.0, 9.0],
        xlabel=r"$\mathrm{RA}\,(\mathrm{degree})$",
        ylabel=r"$\mathrm{Dec}\,(\mathrm{degree})$",
        gcf=True,
    )
    bovy_plot.bovy_plot(
        trackRADec_leading[:, 0],
        trackRADec_leading[:, 1],
        color=sns.color_palette()[0],
        overplot=True,
    )
    plt.plot(
        radec_sample_trailing[:, 0],
        radec_sample_trailing[:, 1],
        "k.",
        alpha=0.01,
        zorder=0,
    )
    plt.plot(
        radec_sample_leading[:, 0],
        radec_sample_leading[:, 1],
        "k.",
        alpha=0.01,
        zorder=0,
    )
    plt.errorbar(
        pos_radec[:, 0],
        pos_radec[:, 1],
        yerr=pos_radec[:, 2],
        ls="none",
        marker="o",
        color=sns.color_palette()[2],
    )
    plt.subplot(1, 2, 2)
    bovy_plot.bovy_plot(
        trackRADec_trailing[:, 0],
        sdf_trailing._interpolatedObsTrackLB[:, 3],
        color=sns.color_palette()[0],
        xrange=[250.0, 210.0],
        yrange=[-80.0, 0.0],
        xlabel=r"$\mathrm{RA}\,(\mathrm{degree})$",
        ylabel=r"$V_{\mathrm{los}}\,(\mathrm{km\,s}^{-1})$",
        gcf=True,
    )
    bovy_plot.bovy_plot(
        trackRADec_leading[:, 0],
        sdf_leading._interpolatedObsTrackLB[:, 3],
        color=sns.color_palette()[0],
        overplot=True,
    )
    plt.plot(
        radec_sample_trailing[:, 0], lb_sample_trailing[3], "k.", alpha=0.01, zorder=0,
    )
    plt.plot(
        radec_sample_leading[:, 0], lb_sample_leading[3], "k.", alpha=0.01, zorder=0,
    )
    plt.errorbar(
        rvel_ra[:, 0],
        rvel_ra[:, 1],
        yerr=rvel_ra[:, 2],
        ls="none",
        marker="o",
        color=sns.color_palette()[2],
    )

    plt.savefig("figures/fiducial_model.pdf")

    return


# /def


def model_vary_c_along_the_best_fit_line():

    bf_savefilename = _MW_POT_SCRIPT_FOLDER + "/output/mwpot14varyc-bf.pkl"
    if os.path.exists(bf_savefilename):
        with open(bf_savefilename, "rb") as savefile:
            cs = pickle.load(savefile)
            bf_params = pickle.load(savefile)
    else:
        IOError(
            (
                "Need to calculate best-fit potentials for different c "
                "in MWPotential2014-varyc.ipynb first"
            )
        )
    bf_params = np.array(bf_params)
    bf_params = bf_params[cs <= 3.0]
    cs = cs[cs <= 3.0]

    # ----------------------------------------------------------

    progs = []
    progfs = []
    times = np.linspace(0.0, 3.0, 101)

    for bp, c in zip(bf_params, cs):
        pot = MWPotential2014Likelihood.setup_potential(
            bp, c, False, False, REFR0, REFV0
        )
        prog = Orbit(
            [229.018, -0.124, 23.2, -2.296, -2.257, -58.7],
            radec=True,
            ro=REFR0,
            vo=REFV0,
            solarmotion=[-11.1, 24.0, 7.25],
        )
        prog.integrate(times, pot)
        progs.append(prog)
        prog = Orbit(
            [229.018, -0.124, 23.2, -2.296, -2.257, -58.7],
            radec=True,
            ro=REFR0,
            vo=REFV0,
            solarmotion=[-11.1, 24.0, 7.25],
        ).flip()
        prog.integrate(times, pot)
        prog.flip(inplace=True)
        progfs.append(prog)

    # ----------------------------------------------------------
    # Vary $c$ along the best-fit line

    plt.figure(figsize=(12, 4))

    for c, orb, orbf in zip(cs, progs, progfs):
        tc = cmap((c - 0.5) / 2.5)
        plt.subplot(1, 2, 1)
        orb.plot(d1="ra", d2="dec", color=tc, overplot=True)
        orbf.plot(d1="ra", d2="dec", color=tc, overplot=True)
        plt.subplot(1, 2, 2)
        orb.plot(d1="ra", d2="vlos", color=tc, overplot=True)
        orbf.plot(d1="ra", d2="vlos", color=tc, overplot=True)

    plot_data_add_labels(color="k")
    add_colorbar(0.5, 3.0, r"$c$", save_figures=False)
    plt.savefig("figures/varyc_bestfitline.pdf")

    return


# /def


def model_vary_b_for_c_1_pa_0():

    # ----------------------------------------------------------
    # Vary $b$ for $c=1$ (pa=0):

    p_b15 = [
        0.60122692,
        0.36273147,
        -0.97591502,
        -3.34169377,
        0.71877924,
        -0.01519337,
        -0.01928001,
    ]
    progs = []
    progfs = []
    times = np.linspace(0.0, 3.0, 101)
    bs = np.arange(0.5, 2.1, 0.1)

    for b in bs:
        pot = MWPotential2014Likelihood.setup_potential(
            p_b15, 1.0, False, False, REFR0, REFV0, b=b
        )
        prog = Orbit(
            [229.018, -0.124, 23.2, -2.296, -2.257, -58.7],
            radec=True,
            ro=REFR0,
            vo=REFV0,
            solarmotion=[-11.1, 24.0, 7.25],
        )
        prog.integrate(times, pot)
        progs.append(prog)
        prog = Orbit(
            [229.018, -0.124, 23.2, -2.296, -2.257, -58.7],
            radec=True,
            ro=REFR0,
            vo=REFV0,
            solarmotion=[-11.1, 24.0, 7.25],
        ).flip()
        prog.integrate(times, pot)
        prog.flip(inplace=True)
        progfs.append(prog)

    # -------------------------------------------

    plt.figure(figsize=(12, 4))

    for b, orb, orbf in zip(bs, progs, progfs):
        tc = cmap((b - 0.5) / 1.5)
        plt.subplot(1, 2, 1)
        orb.plot(d1="ra", d2="dec", color=tc, overplot=True)
        orbf.plot(d1="ra", d2="dec", color=tc, overplot=True)
        plt.subplot(1, 2, 2)
        orb.plot(d1="ra", d2="vlos", color=tc, overplot=True)
        orbf.plot(d1="ra", d2="vlos", color=tc, overplot=True)

    plot_data_add_labels(color="k")
    add_colorbar(0.5, 2.0, r"$b$", save_figures=False)
    plt.savefig("figures/varyb_c-1_pa-0.pdf")

    return


# /def


def model_vary_b_for_c_1_pa_45():

    # ----------------------------------------------------------
    # Vary $b$ for $c=1$ (pa=45 degree):

    p_b15 = [
        0.60122692,
        0.36273147,
        -0.97591502,
        -3.34169377,
        0.71877924,
        -0.01519337,
        -0.01928001,
    ]
    progs = []
    progfs = []
    times = np.linspace(0.0, 3.0, 101)
    bs = np.arange(0.5, 2.1, 0.1)
    for b in bs:
        pot = MWPotential2014Likelihood.setup_potential(
            p_b15, 1.0, False, False, REFR0, REFV0, b=b, pa=np.pi / 4.0
        )
        prog = Orbit(
            [229.018, -0.124, 23.2, -2.296, -2.257, -58.7],
            radec=True,
            ro=REFR0,
            vo=REFV0,
            solarmotion=[-11.1, 24.0, 7.25],
        )
        prog.integrate(times, pot)
        progs.append(prog)
        prog = Orbit(
            [229.018, -0.124, 23.2, -2.296, -2.257, -58.7],
            radec=True,
            ro=REFR0,
            vo=REFV0,
            solarmotion=[-11.1, 24.0, 7.25],
        ).flip()
        prog.integrate(times, pot)
        prog.flip(inplace=True)
        progfs.append(prog)

    # -------------------------------------------

    plt.figure(figsize=(12, 4))

    for b, orb, orbf in zip(bs, progs, progfs):
        tc = cmap((b - 0.5) / 1.5)
        plt.subplot(1, 2, 1)
        orb.plot(d1="ra", d2="dec", color=tc, overplot=True)
        orbf.plot(d1="ra", d2="dec", color=tc, overplot=True)
        plt.subplot(1, 2, 2)
        orb.plot(d1="ra", d2="vlos", color=tc, overplot=True)
        orbf.plot(d1="ra", d2="vlos", color=tc, overplot=True)

    plot_data_add_labels(color="k")
    add_colorbar(0.5, 2.0, r"$b$", save_figures=False)
    plt.savefig("figures/varyb_c-1_pa-45.pdf")

    return


# /def


def model_vary_d():

    # ----------------------------------------------------------
    # The Pal 5 stream is not very sensitive to changes in $b$.
    # Vary the distance:

    p_b15 = [
        0.60122692,
        0.36273147,
        -0.97591502,
        -3.34169377,
        0.71877924,
        -0.01519337,
        -0.01928001,
    ]
    progs = []
    progfs = []
    times = np.linspace(0.0, 3.0, 101)
    ds = np.linspace(22.5, 24.5, 101)
    for d in ds:
        pot = MWPotential2014Likelihood.setup_potential(
            p_b15, 1.0, False, False, REFR0, REFV0
        )
        prog = Orbit(
            [229.018, -0.124, d, -2.296, -2.257, -58.7],
            radec=True,
            ro=REFR0,
            vo=REFV0,
            solarmotion=[-11.1, 24.0, 7.25],
        )
        prog.integrate(times, pot)
        progs.append(prog)
        prog = Orbit(
            [229.018, -0.124, d, -2.296, -2.257, -58.7],
            radec=True,
            ro=REFR0,
            vo=REFV0,
            solarmotion=[-11.1, 24.0, 7.25],
        ).flip()
        prog.integrate(times, pot)
        prog.flip(inplace=True)
        progfs.append(prog)

    # -------------------------------------------

    plt.figure(figsize=(12, 4))

    for d, orb, orbf in zip(ds, progs, progfs):
        tc = cmap((d - np.amin(ds)) / (np.amax(ds) - np.amin(ds)))
        plt.subplot(1, 2, 1)
        orb.plot(d1="ra", d2="dec", color=tc, overplot=True)
        orbf.plot(d1="ra", d2="dec", color=tc, overplot=True)
        plt.subplot(1, 2, 2)
        orb.plot(d1="ra", d2="vlos", color=tc, overplot=True)
        orbf.plot(d1="ra", d2="vlos", color=tc, overplot=True)
    plot_data_add_labels(color="k")
    add_colorbar(
        np.amin(ds),
        np.amax(ds),
        r"$\mathrm{distance}\,(\mathrm{kpc})$",
        save_figures=False,
    )
    plt.savefig("figures/varyd.pdf")


def model_vary_pm_pll():

    # ----------------------------------------------------------
    # Vary the proper motion parallel to the stream

    p_b15 = [
        0.60122692,
        0.36273147,
        -0.97591502,
        -3.34169377,
        0.71877924,
        -0.01519337,
        -0.01928001,
    ]
    progs = []
    progfs = []
    times = np.linspace(0.0, 2.5, 101)
    pms = np.linspace(-0.3, 0.3, 101)
    for pm in pms:
        pot = MWPotential2014Likelihood.setup_potential(
            p_b15, 1.0, False, False, REFR0, REFV0
        )
        prog = Orbit(
            [229.018, -0.124, 23.2, -2.296 + pm, -2.257 + 2.257 / 2.296 * pm, -58.7,],
            radec=True,
            ro=REFR0,
            vo=REFV0,
            solarmotion=[-11.1, 24.0, 7.25],
        )
        prog.integrate(times, pot)
        progs.append(prog)
        prog = Orbit(
            [229.018, -0.124, 23.2, -2.296 + pm, -2.257 + 2.257 / 2.296 * pm, -58.7,],
            radec=True,
            ro=REFR0,
            vo=REFV0,
            solarmotion=[-11.1, 24.0, 7.25],
        ).flip()
        prog.integrate(times, pot)
        # prog._orb.orbit[:,1]*= -1.
        # prog.orbit[:,1]*= -1.
        # prog.orbit[:,2]*= -1.
        # prog.orbit[:,4]*= -1.
        prog.flip(inplace=True)
        progfs.append(prog)

    plt.figure(figsize=(12, 4))

    for pm, orb, orbf in zip(pms, progs, progfs):
        tc = cmap((pm - np.amin(pms)) / (np.amax(pms) - np.amin(pms)))
        plt.subplot(1, 2, 1)
        orb.plot(d1="ra", d2="dec", color=tc, overplot=True)
        orbf.plot(d1="ra", d2="dec", color=tc, overplot=True)
        plt.subplot(1, 2, 2)
        orb.plot(d1="ra", d2="vlos", color=tc, overplot=True)
        orbf.plot(d1="ra", d2="vlos", color=tc, overplot=True)
    plot_data_add_labels(color="k")
    add_colorbar(
        np.amin(pms),
        np.amax(pms),
        r"$\mathrm{proper\ motion\ offset}\,(\mathrm{mas\,yr}^{-1})$",
        save_figures=False,
    )
    plt.savefig("figures/varypm-prll.pdf")


def model_vary_pm_perp():

    # ----------------------------------------------------------
    # Vary the proper motion perpendicular to the stream:

    p_b15 = [
        0.60122692,
        0.36273147,
        -0.97591502,
        -3.34169377,
        0.71877924,
        -0.01519337,
        -0.01928001,
    ]
    progs = []
    progfs = []
    times = np.linspace(0.0, 2.5, 101)
    pms = np.linspace(-0.3, 0.3, 101)
    for pm in pms:
        pot = MWPotential2014Likelihood.setup_potential(
            p_b15, 1.0, False, False, REFR0, REFV0
        )
        prog = Orbit(
            [229.018, -0.124, 23.2, -2.296 + pm, -2.257 - 2.296 / 2.257 * pm, -58.7,],
            radec=True,
            ro=REFR0,
            vo=REFV0,
            solarmotion=[-11.1, 24.0, 7.25],
        )
        prog.integrate(times, pot)
        progs.append(prog)
        prog = Orbit(
            [229.018, -0.124, 23.2, -2.296 + pm, -2.257 - 2.296 / 2.257 * pm, -58.7,],
            radec=True,
            ro=REFR0,
            vo=REFV0,
            solarmotion=[-11.1, 24.0, 7.25],
        ).flip()
        prog.integrate(times, pot)
        prog.flip(inplace=True)
        progfs.append(prog)

    plt.figure(figsize=(12, 4))
    for pm, orb, orbf in zip(pms, progs, progfs):
        tc = cmap((pm - np.amin(pms)) / (np.amax(pms) - np.amin(pms)))
        plt.subplot(1, 2, 1)
        orb.plot(d1="ra", d2="dec", color=tc, overplot=True)
        orbf.plot(d1="ra", d2="dec", color=tc, overplot=True)
        plt.subplot(1, 2, 2)
        orb.plot(d1="ra", d2="vlos", color=tc, overplot=True)
        orbf.plot(d1="ra", d2="vlos", color=tc, overplot=True)
    plot_data_add_labels(color="k")
    add_colorbar(
        np.amin(pms),
        np.amax(pms),
        r"$\mathrm{proper\ motion\ offset}\,(\mathrm{mas\,yr}^{-1})$",
        save_figures=False,
    )
    plt.savefig("figures/varypm-perp.pdf")


#################################################################
# How does the track, width, and length of the Pal 5 stream vary with the
# potential and the uncertain phase-space location of the Pal 5 cluster?


def _get_pal5varyc():

    p_b15 = [
        0.60122692,
        0.36273147,
        -0.97591502,
        -3.34169377,
        0.71877924,
        -0.01519337,
        -0.01928001,
    ]
    savefilename = _MW_POT_SCRIPT_FOLDER + "output/mwpot14-pal5-varyc.pkl"
    if os.path.exists(savefilename):
        with open(savefilename, "rb") as savefile:
            cs = pickle.load(savefile)
            pal5varyc = pickle.load(savefile)
    else:
        cs = np.arange(0.5, 2.3, 0.1)
        pal5varyc = pal5_util.predict_pal5obs(
            p_b15,
            cs,
            multi=8,
            useTM=False,
            interpcs=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25],
        )
        save_pickles(savefilename, cs, pal5varyc)

    return pal5varyc, cs


# /def


def model_stream_struct_vary_c():

    # ----------------------------------------------------------
    # Varying the flattening $c$ of the halo
    # We compute the stream structure for a fiducial set of parameters, to get
    # a sense of where the track lies and how the width and length vary

    pal5varyc, cs = _get_pal5varyc()

    # -------------------------------------------

    plt.figure(figsize=(12, 4))
    for ii, c in enumerate(cs):
        tc = cmap((c - np.amin(cs)) / (np.amax(cs) - np.amin(cs)))
        plt.subplot(1, 2, 1)
        bovy_plot.bovy_plot(
            pal5varyc[0][ii, :, 0], pal5varyc[0][ii, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varyc[1][ii, :, 0], pal5varyc[1][ii, :, 1], color=tc, overplot=True,
        )
        plt.subplot(1, 2, 2)
        bovy_plot.bovy_plot(
            pal5varyc[2][ii, :, 0], pal5varyc[2][ii, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varyc[3][ii, :, 0], pal5varyc[3][ii, :, 1], color=tc, overplot=True,
        )
    plot_data_add_labels()
    plt.tight_layout()
    plt.savefig("figures/mwpot14-pal5-varyc.pdf")

    return


# /def


def _get_pal5varyvc():

    pal5varyc, cs = _get_pal5varyc()

    savefilename = _MW_POT_SCRIPT_FOLDER + "output/mwpot14-pal5-varyvc.pkl"

    p_b15 = [
        0.60122692,
        0.36273147,
        -0.97591502,
        -3.34169377,
        0.71877924,
        -0.01519337,
        -0.01928001,
    ]
    if os.path.exists(savefilename):
        with open(savefilename, "rb") as savefile:
            vcs = pickle.load(savefile)
            pal5varyvc = pickle.load(savefile)
    else:
        vcs = np.arange(200.0, 255.0, 5.0)
        pal5varyvc = np.zeros(
            (len(vcs), 4, pal5varyc[0].shape[1], pal5varyc[0].shape[2])
        )
        for ii, vc in enumerate(vcs):
            t = pal5_util.predict_pal5obs(
                p_b15, 1.0, multi=8, vo=vc, singlec=True, useTM=False
            )
            for jj in range(4):
                pal5varyvc[ii, jj] = t[jj][0]
        save_pickles(savefilename, vcs, pal5varyvc)

    return pal5varyvc, vcs


def model_stream_struct_vary_vcirc():
    # ----------------------------------------------------------
    # Varying the circular velocity $V_c(R_0)$ to change the normalization of the potential

    pal5varyvc, vcs = _get_pal5varyc()

    plt.figure(figsize=(12, 4))

    for ii, vc in enumerate(vcs):
        tc = cmap((vc - np.amin(vcs)) / (np.amax(vcs) - np.amin(vcs)))
        plt.subplot(1, 2, 1)
        bovy_plot.bovy_plot(
            pal5varyvc[ii, 0, :, 0], pal5varyvc[ii, 0, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varyvc[ii, 1, :, 0], pal5varyvc[ii, 1, :, 1], color=tc, overplot=True,
        )
        plt.subplot(1, 2, 2)
        bovy_plot.bovy_plot(
            pal5varyvc[ii, 2, :, 0], pal5varyvc[ii, 2, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varyvc[ii, 3, :500, 0],
            pal5varyvc[ii, 3, :500, 1],
            color=tc,
            overplot=True,
        )
    plot_data_add_labels()
    plt.tight_layout()
    plt.savefig("figures/mwpot14-pal5-varyvc.pdf")


def _get_pal5varyd():

    pal5varyc, cs = _get_pal5varyc()

    savefilename = _MW_POT_SCRIPT_FOLDER + "output/mwpot14-pal5-varyd.pkl"

    p_b15 = [
        0.60122692,
        0.36273147,
        -0.97591502,
        -3.34169377,
        0.71877924,
        -0.01519337,
        -0.01928001,
    ]

    if os.path.exists(savefilename):
        with open(savefilename, "rb") as savefile:
            ds = pickle.load(savefile)
            pal5varyd = pickle.load(savefile)
    else:
        ds = np.arange(20.0, 24.5, 0.5)
        pal5varyd = np.zeros((len(ds), 4, pal5varyc[0].shape[1], pal5varyc[0].shape[2]))
        for ii, d in enumerate(tqdm(ds)):
            t = pal5_util.predict_pal5obs(
                p_b15, 1.0, multi=8, dist=d, singlec=True, useTM=False
            )
            for jj in range(4):
                pal5varyd[ii, jj] = t[jj][0]
        save_pickles(savefilename, ds, pal5varyd)

    return pal5varyd, ds


def model_stream_struct_vary_d():

    pal5varyc, cs = _get_pal5varyc()
    pal5varyd, ds = _get_pal5varyd()

    # ----------------------------------------------------------
    # Varying the distance to the Pal 5 cluster

    plt.figure(figsize=(12, 4))

    for ii, d in enumerate(ds):
        tc = cmap((d - np.amin(ds)) / (np.amax(ds) - np.amin(ds)))
        plt.subplot(1, 2, 1)
        bovy_plot.bovy_plot(
            pal5varyd[ii, 0, :, 0], pal5varyd[ii, 0, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varyd[ii, 1, :, 0], pal5varyd[ii, 1, :, 1], color=tc, overplot=True,
        )
        plt.subplot(1, 2, 2)
        bovy_plot.bovy_plot(
            pal5varyd[ii, 2, :, 0], pal5varyd[ii, 2, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varyd[ii, 3, :500, 0],
            pal5varyd[ii, 3, :500, 1],
            color=tc,
            overplot=True,
        )
    plot_data_add_labels()
    plt.tight_layout()
    plt.savefig("figures/mwpot14-pal5-varyd.pdf")


def _get_pal5varypm():

    pal5varyc, cs = _get_pal5varyc()

    savefilename = _MW_POT_SCRIPT_FOLDER + "output/mwpot14-pal5-varypm.pkl"

    p_b15 = [
        0.60122692,
        0.36273147,
        -0.97591502,
        -3.34169377,
        0.71877924,
        -0.01519337,
        -0.01928001,
    ]

    if os.path.exists(savefilename):
        with open(savefilename, "rb") as savefile:
            pms = pickle.load(savefile)
            pal5varypm = pickle.load(savefile)
    else:
        pms = np.arange(-0.3, 0.35, 0.05)
        pal5varypm = np.zeros(
            (len(pms), 4, pal5varyc[0].shape[1], pal5varyc[0].shape[2])
        )
        for ii, pm in enumerate(tqdm(pms)):
            pmra, pmdec = -2.296 + pm, -2.257 + 2.257 / 2.296 * pm
            t = pal5_util.predict_pal5obs(
                p_b15, 1.0, multi=8, pmra=pmra, pmdec=pmdec, singlec=True, useTM=False,
            )
            for jj in range(4):
                pal5varypm[ii, jj] = t[jj][0]
        save_pickles(savefilename, pms, pal5varypm)

    return pal5varypm, pms


# /def


def model_stream_struct_vary_pm():

    pal5varyc, cs = _get_pal5varyc()
    pal5varypm, pms = _get_pal5varypm()

    # ----------------------------------------------------------
    # Varying the proper motion of the Pal 5 cluster

    plt.figure(figsize=(12, 4))

    for ii, pm in enumerate(pms):
        tc = cmap((pm - np.amin(pms)) / (np.amax(pms) - np.amin(pms)))
        plt.subplot(1, 2, 1)
        bovy_plot.bovy_plot(
            pal5varypm[ii, 0, :, 0], pal5varypm[ii, 0, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varypm[ii, 1, :, 0], pal5varypm[ii, 1, :, 1], color=tc, overplot=True,
        )
        plt.subplot(1, 2, 2)
        bovy_plot.bovy_plot(
            pal5varypm[ii, 2, :, 0], pal5varypm[ii, 2, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varypm[ii, 3, :500, 0],
            pal5varypm[ii, 3, :500, 1],
            color=tc,
            overplot=True,
        )
    plot_data_add_labels()
    plt.tight_layout()
    plt.savefig("figures/mwpot14-pal5-varypm.pdf")


def plot_all():

    pal5varyc, cs = _get_pal5varyc()
    pal5varyd, ds = _get_pal5varyd()
    pal5varypm, pms = _get_pal5varypm()

    # ----------------------------------------------------------
    # All in one plot

    bovy_plot.bovy_print(
        axes_labelsize=17.0,
        text_fontsize=12.0,
        xtick_labelsize=15.0,
        ytick_labelsize=15.0,
    )
    plt.figure(figsize=(16, 7))

    gs = gridspec.GridSpec(
        3, 4, wspace=0.1, hspace=0.075, height_ratios=[0.075, 1.0, 1.0]
    )
    # c
    for ii, c in enumerate(cs):
        tc = cmap((c - np.amin(cs)) / (np.amax(cs) - np.amin(cs)))
        plt.subplot(gs[4])
        bovy_plot.bovy_plot(
            pal5varyc[0][ii, :, 0], pal5varyc[0][ii, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varyc[1][ii, :, 0], pal5varyc[1][ii, :, 1], color=tc, overplot=True,
        )
        plt.subplot(gs[8])
        bovy_plot.bovy_plot(
            pal5varyc[2][ii, :, 0], pal5varyc[2][ii, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varyc[3][ii, :, 0], pal5varyc[3][ii, :, 1], color=tc, overplot=True,
        )
    plot_data_add_labels(p1=(gs[4],), p2=(gs[8],), noxlabel_dec=True)
    # Vc
    for ii, vc in enumerate(vcs):
        tc = cmap((vc - np.amin(vcs)) / (np.amax(vcs) - np.amin(vcs)))
        plt.subplot(gs[5])
        bovy_plot.bovy_plot(
            pal5varyvc[ii, 0, :, 0], pal5varyvc[ii, 0, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varyvc[ii, 1, :, 0], pal5varyvc[ii, 1, :, 1], color=tc, overplot=True,
        )
        plt.subplot(gs[9])
        bovy_plot.bovy_plot(
            pal5varyvc[ii, 2, :, 0], pal5varyvc[ii, 2, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varyvc[ii, 3, :500, 0],
            pal5varyvc[ii, 3, :500, 1],
            color=tc,
            overplot=True,
        )
    plot_data_add_labels(p1=(gs[5],), p2=(gs[9],), noylabel=True, noxlabel_dec=True)
    # D
    for ii, d in enumerate(ds):
        tc = cmap((d - np.amin(ds)) / (np.amax(ds) - np.amin(ds)))
        plt.subplot(gs[6])
        bovy_plot.bovy_plot(
            pal5varyd[ii, 0, :, 0], pal5varyd[ii, 0, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varyd[ii, 1, :, 0], pal5varyd[ii, 1, :, 1], color=tc, overplot=True,
        )
        plt.subplot(gs[10])
        bovy_plot.bovy_plot(
            pal5varyd[ii, 2, :, 0], pal5varyd[ii, 2, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varyd[ii, 3, :500, 0],
            pal5varyd[ii, 3, :500, 1],
            color=tc,
            overplot=True,
        )
    plot_data_add_labels(p1=(gs[6],), p2=(gs[10],), noylabel=True, noxlabel_dec=True)
    # PM
    for ii, pm in enumerate(pms):
        tc = cmap((pm - np.amin(pms)) / (np.amax(pms) - np.amin(pms)))
        plt.subplot(gs[7])
        bovy_plot.bovy_plot(
            pal5varypm[ii, 0, :, 0], pal5varypm[ii, 0, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varypm[ii, 1, :, 0], pal5varypm[ii, 1, :, 1], color=tc, overplot=True,
        )
        plt.subplot(gs[11])
        bovy_plot.bovy_plot(
            pal5varypm[ii, 2, :, 0], pal5varypm[ii, 2, :, 1], color=tc, overplot=True,
        )
        bovy_plot.bovy_plot(
            pal5varypm[ii, 3, :500, 0],
            pal5varypm[ii, 3, :500, 1],
            color=tc,
            overplot=True,
        )
    plot_data_add_labels(p1=(gs[7],), p2=(gs[11],), noylabel=True, noxlabel_dec=True)
    # Colorbars
    for ii, (label, vals, ticks) in enumerate(
        zip(
            [
                r"$\mathrm{halo\ axis\ ratio}$",
                r"$V_c\,(\mathrm{km\,s}^{-1})$",
                r"$\mathrm{Distance}\,(\mathrm{kpc})$",
                r"$\Delta\mu_{\parallel}\,(\mathrm{mas\,yr}^{-1})$",
            ],
            [cs, vcs, ds, pms * np.sqrt(1.0 + (2.257 / 2.296) ** 2.0)],
            [
                [0.5, 1.0, 1.5, 2.0],
                [200.0, 210.0, 220.0, 230.0, 240.0, 250.0],
                [20.0, 21, 22, 23, 24],
                [-0.4, -0.2, 0.0, 0.2, 0.4],
            ],
        )
    ):
        plt.subplot(gs[ii])
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=np.amin(vals), vmax=np.amax(vals)),
        )
        sm._A = []
        CB1 = plt.colorbar(sm, orientation="horizontal", cax=plt.gca(), ticks=ticks)
        CB1.set_label(label, fontsize=16.0)
        CB1.ax.xaxis.set_ticks_position("top")
        CB1.ax.xaxis.set_label_position("top")
    if save_figures:
        plt.savefig(
            os.path.join(os.getenv("PAPERSDIR"), "2016-mwhalo-shape", "pal5track.pdf"),
            bbox_inches="tight",
        )
    plt.savefig("figures/2016-mwhalo-shape-pal5track.pdf")


###############################################################################
# Command Line
###############################################################################


def make_parser(inheritable=False):
    """Expose parser for ``main``.

    Parameters
    ----------
    inheritable: bool
        whether the parser can be inherited from (default False).
        if True, sets ``add_help=False`` and ``conflict_hander='resolve'``

    Returns
    -------
    parser: ArgumentParser

    """
    parser = argparse.ArgumentParser(
        description="",
        add_help=~inheritable,
        conflict_handler="resolve" if ~inheritable else "error",
    )

    return parser


# /def


# ------------------------------------------------------------------------


def main(args: Optional[list] = None, opts: Optional[argparse.Namespace] = None):
    """Script Function.

    Parameters
    ----------
    args : list, optional
        an optional single argument that holds the sys.argv list,
        except for the script name (e.g., argv[1:])
    opts : Namespace, optional
        pre-constructed results of parsed args
        if not None, used ONLY if args is None

    """
    if opts is not None and args is None:
        pass
    else:
        if opts is not None:
            warnings.warn("Not using `opts` because `args` are given")
        parser = make_parser()
        opts = parser.parse_args(args)

    return


# /def


# ------------------------------------------------------------------------

if __name__ == "__main__":

    main(args=None, opts=None)  # all arguments except script name

# /if


###############################################################################
# END
