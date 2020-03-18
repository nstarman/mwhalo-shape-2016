# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   :
# AUTHOR  :
# PROJECT :
#
# ----------------------------------------------------------------------------

# Docstring
"""Script Utilities.

Routing Listings
----------------
fit
sample
sample_multi
plot_samples
plot_mcmc_c

"""

__author__ = "Jo Bovy"
__maintainer__ = "Nathaniel Starkman"

# __all__ = [
#     ""
# ]


###############################################################################
# IMPORTS

# GENERAL
import os
import os.path
import pickle
import shutil
from typing import Optional

import numpy as np
from tqdm import tqdm

from scipy import special  # , integrate

# MC-related

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import gridspec, cm

# galpy
from galpy.util import bovy_plot, bovy_conversion
from galpy import potential


# PROJECT-SPECIFIC

from ... import mw_pot
from ...mw_pot.utils import fit, sample, sample_multi, plot_samples
from ...mw_pot import MWPotential2014Likelihood
from ...mw_pot.data import (
    readBovyRix13kzdata,
    readClemens,
    readMcClureGriffiths07,
    readMcClureGriffiths16,
)


###############################################################################
# PARAMETERS

np.random.seed(1)  # set random number seed. TODO use numpy1.8 generator

_REFR0 = mw_pot._REFR0
_REFV0 = mw_pot._REFV0


###############################################################################
# CODE
###############################################################################


def make_simulation_folders(drct: str):
    """Make Simulation Folder Structure.

    Makes figures, output, and relevant subfolders, if they do not exist.
        figures/mwpot14
        output/

    Parameters
    ----------
    drct: str
        the simulation folder

    """
    if not drct.endswith("/"):
        drct += "/"

    dirs = (
        # figures
        drct + "figures",
        drct + "figures/mwpot14",
        drct + "figures/mwpot_dblexp",
        # output
        drct + "output",
    )

    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path)

    return

# /def


# --------------------------------------------------------------------------


def clear_simulation(drct: str, clear_output: bool = True):
    """Clear Simulation Folder Structure.

    Dos NOT clear the output folder

    """
    if not drct.endswith("/"):
        drct += "/"

    dirs = [
        drct + "figures",
    ]
    if clear_output:
        dirs.append(drct + "output")

    def _remove_all(path):
        for d in os.listdir(path):
            try:
                os.remove(path + "/" + d)
            except os.IsADirectoryError:
                _remove_all(path + "/" + d)

    for path in dirs:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    make_simulation_folders(drct)

    return


# /def


###############################################################################


def plot_mcmc_c(
    samples, fitvoro, cs, bf_params, add_families=False, savefig=None
):
    """Plot MCMC C-Parameter.

    Parameters
    ----------
    samples : list
    fitvoro : bool
    add_families : bool, optional

    Returns
    -------
    None

    """
    if add_families:
        st0 = np.random.get_state()
        np.random.seed(1)
        with open("output/mwpot14varyc-samples.pkl", "rb") as savefile:
            pot_samples = pickle.load(savefile)
        rndindices = np.random.permutation(pot_samples.shape[1])
        pot_params = np.zeros((8, 32))
        for ii in range(32):
            pot_params[:, ii] = pot_samples[:, rndindices[ii]]

    cindx = 9 if fitvoro else 7

    cmap = cm.viridis
    levels = list(special.erf(np.arange(1, 3) / np.sqrt(2.0)))
    levels.append(1.01)

    def axes_white():
        for k, spine in plt.gca().spines.items():  # ax.spines is a dictionary
            spine.set_color("w")
        plt.gca().tick_params(axis="x", which="both", colors="w")
        plt.gca().tick_params(axis="y", which="both", colors="w")
        [t.set_color("k") for t in plt.gca().xaxis.get_ticklabels()]
        [t.set_color("k") for t in plt.gca().yaxis.get_ticklabels()]
        return None

    # ------------------------------

    fig = plt.figure(figsize=(16 + 2 * fitvoro, 4))
    gs = gridspec.GridSpec(1, 5 + 2 * fitvoro, wspace=0.015)
    plt.subplot(gs[0])
    bovy_plot.scatterplot(
        samples[0],
        samples[cindx],
        ",",
        gcf=True,
        bins=31,
        cmap=cmap,
        cntrcolors="0.6",
        xrange=[0.0, 0.99],
        yrange=[0.0, 4.0],
        levels=levels,
        xlabel=r"$f_d$",
        ylabel=r"$c/a$",
        zorder=1,
    )
    bovy_plot.bovy_plot([bp[0] for bp in bf_params], cs, "w-", overplot=True)
    if add_families:
        fm_kw = {
            "color": "w",
            "marker": "x",
            "ls": "none",
            "ms": 4.0,
            "mew": 1.2,
        }
        bovy_plot.bovy_plot(
            pot_params[0], pot_params[7], overplot=True, **fm_kw
        )
    axes_white()
    plt.subplot(gs[1])
    bovy_plot.scatterplot(
        samples[1],
        samples[cindx],
        ",",
        gcf=True,
        bins=31,
        cmap=cmap,
        cntrcolors="0.6",
        xrange=[0.0, 0.99],
        yrange=[0.0, 4.0],
        levels=levels,
        xlabel=r"$f_h$",
    )
    bovy_plot.bovy_plot([bp[1] for bp in bf_params], cs, "w-", overplot=True)
    if add_families:
        bovy_plot.bovy_plot(
            pot_params[1], pot_params[7], overplot=True, **fm_kw
        )
    nullfmt = NullFormatter()
    plt.gca().yaxis.set_major_formatter(nullfmt)
    axes_white()
    plt.subplot(gs[2])
    bovy_plot.scatterplot(
        np.exp(samples[2]) * _REFR0,
        samples[cindx],
        ",",
        gcf=True,
        bins=21,
        levels=levels,
        xrange=[2.0, 4.49],
        yrange=[0.0, 4.0],
        cmap=cmap,
        cntrcolors="0.6",
        xlabel=r"$h_R\,(\mathrm{kpc})$",
    )
    bovy_plot.bovy_plot(
        [np.exp(bp[2]) * _REFR0 for bp in bf_params], cs, "w-", overplot=True,
    )
    if add_families:
        bovy_plot.bovy_plot(
            np.exp(pot_params[2]) * _REFR0,
            pot_params[7],
            overplot=True,
            **fm_kw,
        )
    nullfmt = NullFormatter()
    plt.gca().yaxis.set_major_formatter(nullfmt)
    axes_white()
    plt.subplot(gs[3])
    bovy_plot.scatterplot(
        np.exp(samples[3]) * _REFR0 * 1000.0,
        samples[cindx],
        ",",
        gcf=True,
        bins=26,
        levels=levels,
        xrange=[150.0, 445.0],
        yrange=[0.0, 4.0],
        cmap=cmap,
        cntrcolors="0.6",
        xlabel=r"$h_z\,(\mathrm{pc})$",
    )
    bovy_plot.bovy_plot(
        [np.exp(bp[3]) * _REFR0 * 1000.0 for bp in bf_params],
        cs,
        "w-",
        overplot=True,
    )
    if add_families:
        bovy_plot.bovy_plot(
            np.exp(pot_params[3]) * _REFR0 * 1000.0,
            pot_params[7],
            overplot=True,
            **fm_kw,
        )
    nullfmt = NullFormatter()
    plt.gca().yaxis.set_major_formatter(nullfmt)
    plt.gca().xaxis.set_ticks([200.0, 300.0, 400.0])
    axes_white()
    plt.subplot(gs[4])
    bovy_plot.scatterplot(
        np.exp(samples[4]) * _REFR0,
        samples[cindx],
        ",",
        gcf=True,
        bins=26,
        levels=levels,
        xrange=[0.0, 39.0],
        yrange=[0.0, 4.0],
        cmap=cmap,
        cntrcolors="0.6",
        xlabel=r"$r_s\,(\mathrm{kpc})$",
    )
    bovy_plot.bovy_plot(
        [np.exp(bp[4]) * _REFR0 for bp in bf_params], cs, "w-", overplot=True,
    )
    if add_families:
        bovy_plot.bovy_plot(
            np.exp(pot_params[4]) * _REFR0,
            pot_params[7],
            overplot=True,
            **fm_kw,
        )
    nullfmt = NullFormatter()
    plt.gca().yaxis.set_major_formatter(nullfmt)
    axes_white()

    if fitvoro:
        plt.subplot(gs[5])
        bovy_plot.scatterplot(
            samples[7] * _REFR0,
            samples[cindx],
            ",",
            gcf=True,
            bins=26,
            levels=levels,
            xrange=[7.1, 8.9],
            yrange=[0.0, 4.0],
            cmap=cmap,
            cntrcolors="0.6",
            xlabel=r"$R_0\,(\mathrm{kpc})$",
        )
        nullfmt = NullFormatter()
        plt.gca().yaxis.set_major_formatter(nullfmt)
        axes_white()
        plt.subplot(gs[6])
        bovy_plot.scatterplot(
            samples[8] * _REFV0,
            samples[cindx],
            ",",
            gcf=True,
            bins=26,
            levels=levels,
            xrange=[200.0, 250.0],
            yrange=[0.0, 4.0],
            cmap=cmap,
            cntrcolors="0.6",
            xlabel=r"$V_c(R_0)\,(\mathrm{km\,s}^{-1})$",
        )
        nullfmt = NullFormatter()
        plt.gca().yaxis.set_major_formatter(nullfmt)
        axes_white()

    if savefig is not None:
        fig.savefig(savefig)

    return None


# /def


# -------------------------------------------------------------------------


def plotForceField(savefig: Optional[str] = None):
    """Plot MW Force Field.

    Parameters
    ----------
    savefic: str, optional

    """
    p_b15_pal5gd1_voro = fit(
        fitc=True, c=None, addpal5=True, addgd1=True, fitvoro=True, mc16=True
    )

    # Set up potential
    p_b15 = p_b15_pal5gd1_voro[0]
    ro, vo = _REFR0, _REFV0
    pot = MWPotential2014Likelihood.setup_potential(
        p_b15, p_b15_pal5gd1_voro[0][-1], False, False, ro, vo
    )
    # Compute force field
    Rs = np.linspace(0.01, 20.0, 51)
    zs = np.linspace(-20.0, 20.0, 151)
    mRs, mzs = np.meshgrid(Rs, zs, indexing="ij")
    forces = np.zeros((len(Rs), len(zs), 2))
    potvals = np.zeros((len(Rs), len(zs)))
    for ii in tqdm(range(len(Rs))):
        for jj in tqdm(range(len(zs))):
            forces[ii, jj, 0] = potential.evaluateRforces(
                pot,
                mRs[ii, jj] / ro,
                mzs[ii, jj] / ro,
                use_physical=True,
                ro=ro,
                vo=vo,
            )
            forces[ii, jj, 1] = potential.evaluatezforces(
                pot,
                mRs[ii, jj] / ro,
                mzs[ii, jj] / ro,
                use_physical=True,
                ro=ro,
                vo=vo,
            )
            potvals[ii, jj] = potential.evaluatePotentials(
                pot,
                mRs[ii, jj] / ro,
                mzs[ii, jj] / ro,
                use_physical=True,
                ro=ro,
                vo=vo,
            )

    fig = plt.figure(figsize=(8, 16))
    skip = 10  # Make sure to keep zs symmetric!!
    scale = 35.0
    # Don't plot these
    # forces[(mRs < 5.)*(np.fabs(mzs) < 4.)]= np.nan
    forces[(mRs < 2.0) * (np.fabs(mzs) < 5.0)] = np.nan
    bovy_plot.bovy_dens2d(
        potvals.T,
        origin="lower",
        cmap="viridis",
        xrange=[Rs[0], Rs[-1]],
        yrange=[zs[0], zs[-1]],
        xlabel=r"$R\,(\mathrm{kpc})$",
        ylabel=r"$Z\,(\mathrm{kpc})$",
        contours=True,
        aspect=1.0,
    )
    plt.quiver(
        mRs[1::skip, 5:-1:skip],
        mzs[1::skip, 5:-1:skip],
        forces[1::skip, 5:-1:skip, 0],
        forces[1::skip, 5:-1:skip, 1],
        scale=scale,
    )
    # Add a few lines pointing to the GC
    for angle in tqdm(np.linspace(0.0, np.pi / 2.0, 8)):
        plt.plot(
            (0.0, 100.0 * np.cos(angle)), (0.0, 100.0 * np.sin(angle)), "k:"
        )
        plt.plot(
            (0.0, 100.0 * np.cos(angle)), (0.0, -100.0 * np.sin(angle)), "k:"
        )
    # Add measurements
    # Pal 5
    plt.quiver(
        (8.0,), (16.0,), (-0.8,), (-1.82,), color="w", zorder=10, scale=scale
    )
    # GD-1
    plt.quiver(
        (12.5,),
        (6.675,),
        (-2.51,),
        (-1.47,),
        color="w",
        zorder=10,
        scale=scale,
    )
    # Disk + flat APOGEE rotation curve:
    # Use Bovy & Tremaine (2012) method for translating F_R in the plane to F_R
    # at 1.1 kpc: dFr/dz = dFz / dR
    diskrs = np.linspace(5.5, 8.5, 3)
    diskfzs = (
        -67.0
        * np.exp(-(diskrs - 8.0) / 2.7)
        / bovy_conversion.force_in_2piGmsolpc2(220.0, 8.0)
        * bovy_conversion.force_in_kmsMyr(220.0, 8.0)
    )
    diskfrs = (
        -(218.0 ** 2.0 / diskrs) * bovy_conversion._kmsInPcMyr / 1000.0
        - 1.1 * diskfzs / 2.7
    )
    plt.quiver(
        diskrs,
        1.1 * np.ones_like(diskrs),
        diskfrs,
        diskfzs,
        color="w",
        zorder=10,
        scale=scale,
    )
    # Labels
    bovy_plot.bovy_text(5.8, 16.0, r"$\mathbf{Pal\ 5}$", color="w", size=17.0)
    bovy_plot.bovy_text(12.5, 7.0, r"$\mathbf{GD-1}$", color="w", size=17.0)
    bovy_plot.bovy_text(
        8.65, 0.5, r"$\mathbf{disk\ stars}$", color="w", size=17.0
    )

    if savefig is not None:
        fig.savefig(savefig)

    return fig

# /def


###############################################################################
# END
