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
import numpy as np
import shutil
import functools
from tqdm import tqdm

from scipy import special  # , integrate
from scipy import optimize

# MC-related
import emcee
import corner

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import gridspec, cm

# galpy
from galpy.util import bovy_plot, bovy_conversion
from galpy import potential

# typing
from typing import Optional

# CUSTOM

import bovy_mcmc  # TODO not need

# PROJECT-SPECIFIC
# fmt: off
import sys; sys.path.insert(0, "../../../")
# fmt: on
from src import MWPotential2014Likelihood
from src.data import (
    readBovyRix13kzdata,
    readClemens,
    readMcClureGriffiths07,
    readMcClureGriffiths16,
)


###############################################################################
# PARAMETERS

np.random.seed(1)  # set random number seed. TODO use numpy1.8 generator

_REFR0 = MWPotential2014Likelihood._REFR0
_REFV0 = MWPotential2014Likelihood._REFV0

# -------------------------------------------------------------------------

# Read the necessary data
# First read the surface densities
surfrs, kzs, kzerrs = readBovyRix13kzdata()

# Then the terminal velocities
cl_glon, cl_vterm, cl_corr = readClemens(dsinl=0.125)
mc_glon, mc_vterm, mc_corr = readMcClureGriffiths07(dsinl=0.125, bin=True)
mc16_glon, mc16_vterm, mc16_corr = readMcClureGriffiths16(
    dsinl=0.125, bin=True
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


# /def


###############################################################################


@functools.lru_cache(maxsize=32)
def fit(
    fitc: bool = False,
    ro: float = _REFR0,
    vo: float = _REFV0,
    fitvoro: bool = False,
    c: float = 1.0,
    dblexp: bool = False,
    plots: bool = True,
    addpal5: bool = False,
    addgd1: bool = False,
    mc16: bool = False,
    addgas: bool = False,
):
    """Perform a Fit.

    Parameters
    ----------
    fitc : bool, optional
    ro : float or Quantity, optional
    vo : float or Quantity, optional
    fitvoro : bool, optional
    c : float, optional
    dblexp : bool, optional
    plots: bool, optional
    addpal5: bool, optional
    addgd1: bool, optional
    mc16: bool, optional
    addgas: bool, optional

    Returns
    -------
    params: list
    like_func: float

    """
    init_params = [
        0.5,
        0.45,
        np.log(2.5 / 8.0),
        np.log(0.4 / 8.0),
        np.log(20.0 / 8.0),
        0.0,
        0.0,
    ]
    if fitvoro:
        init_params.extend([1.0, 1.0])
    if fitc:
        init_params.append(1.0)

    if mc16:
        funcargs = (
            c,
            surfrs,
            kzs,
            kzerrs,
            termdata_mc16,
            7.0,
            fitc,
            fitvoro,
            dblexp,
            addpal5,
            addgd1,
            ro,
            vo,
            addgas,
        )
    else:
        funcargs = (
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
        )

    params = optimize.fmin_powell(
        MWPotential2014Likelihood.like_func,
        init_params,
        args=funcargs,
        disp=False,
    )

    ln_like = MWPotential2014Likelihood.like_func(params, *funcargs)

    # ----------------

    if plots:

        if fitvoro:
            ro, vo = _REFR0 * params[8], _REFV0 * params[7]

        pot = MWPotential2014Likelihood.setup_potential(
            params, c, fitc, dblexp, ro, vo, fitvoro=fitvoro
        )

        fig = plt.Figure()
        plt.subplot(1, 3, 1)
        MWPotential2014Likelihood.plotRotcurve(pot)
        plt.subplot(1, 3, 2)
        MWPotential2014Likelihood.plotKz(pot, surfrs, kzs, kzerrs, ro, vo)
        plt.subplot(1, 3, 3)
        if mc16:
            MWPotential2014Likelihood.plotTerm(pot, termdata_mc16, ro, vo)
        else:
            MWPotential2014Likelihood.plotTerm(pot, termdata, ro, vo)

        plt.suptitle(r"p: " + str(params) + r"  $\mathcal{L}$:" + str(ln_like))

        fig.tight_layout()

        if not isinstance(plots, str):
            savefig = (
                f"figures/fit-fitc_{fitc}-fitvoro_{fitvoro}-c_{c}-"
                f"dblexp_{dblexp}-addpal5_{addpal5}-addgd1_{addgd1}-"
                f"mc16_{mc16}-addgas_{addgas}.png"
            )
        else:
            savefig = plots

        plt.savefig(savefig)

    # /if

    return params, ln_like


# /def


# --------------------------------------------------------------------------


def sample(
    nsamples=1000,
    params=None,
    fitc=False,
    ro=_REFR0,
    vo=_REFV0,
    fitvoro=False,
    c=1.0,
    dblexp=False,
    addpal5=False,
    addgd1=False,
    plots=True,
    mc16=False,
    addgas=False,
    _use_emcee=True,
):
    """Sample.

    Parameters
    ----------
    nsamples : int, optional
    params : list, optional
    fitc : bool, optional
    ro : float, optional
    vo : float, optional
    fitvoro : bool, optional
    c : float, optional
    dblexp : bool, optional
    addpal5 : bool, optional
    addgd1 : bool, optional
    plots : bool, optional
    mc16 : bool, optional
    addgas : bool, optional
    _use_emcee : bool, optional

    Returns
    -------
    samples

    """
    if params is None:
        params = fit(
            fitc=fitc,
            ro=ro,
            vo=vo,
            fitvoro=fitvoro,
            c=c,
            dblexp=dblexp,
            plots=False,
            addpal5=addpal5,
            addgd1=addgd1,
            addgas=addgas,
        )[0]
    if mc16:
        funcargs = (
            c,
            surfrs,
            kzs,
            kzerrs,
            termdata_mc16,
            7.0,
            fitc,
            fitvoro,
            dblexp,
            addpal5,
            addgd1,
            ro,
            vo,
            addgas,
        )
    else:
        funcargs = (
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
        )

    samples = bovy_mcmc.markovpy(
        params,
        0.2,
        lambda x: -MWPotential2014Likelihood.like_func(x, *funcargs),
        (),
        isDomainFinite=[[False, False] for ii in range(len(params))],
        domain=[[0.0, 0.0] for ii in range(len(params))],
        nsamples=nsamples,
        nwalkers=2 * len(params),
        _use_emcee=_use_emcee,
    )
    samples = np.array(samples).T

    if plots:
        plot_samples(samples, fitc, fitvoro, ro=ro, vo=vo)

    return samples


# /def


# --------------------------------------------------------------------------


def sample_multi(
    nsamples=1000,
    params=None,
    fitc=False,
    ro=_REFR0,
    vo=_REFV0,
    fitvoro=False,
    c=1.0,
    dblexp=False,
    addpal5=False,
    addgd1=False,
    plots=True,
    mc16=False,
    addgas=False,
):
    """Sample_multi.

    Parameters
    ----------
    nsamples: int, optional
    params: list, optional
    fitc: bool, optional
    ro: float, optional
    vo: float, optional
    fitvoro: bool, optional
    c: float, optional
    dblexp: bool, optional
    addpal5: bool, optional
    addgd1: bool, optional
    plots: bool, optional
    mc16: bool, optional
    addgas: bool, optional

    Returns
    -------
    samples

    """
    if params is None:
        params = fit(
            fitc=fitc,
            ro=ro,
            vo=vo,
            fitvoro=fitvoro,
            c=c,
            dblexp=dblexp,
            plots=False,
            addpal5=addpal5,
            addgd1=addgd1,
            addgas=addgas,
        )[0]

    if mc16:
        funcargs = (
            c,
            surfrs,
            kzs,
            kzerrs,
            termdata_mc16,
            7.0,
            fitc,
            fitvoro,
            dblexp,
            addpal5,
            addgd1,
            ro,
            vo,
            addgas,
        )
    else:
        funcargs = (
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
        )

    nwalkers = 2 * len(params)
    nn = 0
    all_start_params = np.zeros((nwalkers, len(params)))
    start_lnprob0 = np.zeros(nwalkers)
    step = 0.05 * np.ones(len(params))

    while nn < nwalkers:
        all_start_params[nn] = (
            params + np.random.normal(size=len(params)) * step
        )
        start_lnprob0[nn] = MWPotential2014Likelihood.pdf_func(
            all_start_params[nn], *funcargs
        )
        if start_lnprob0[nn] > -1000000.0:
            nn += 1

    sampler = emcee.EnsembleSampler(
        nwalkers,
        len(params),
        MWPotential2014Likelihood.pdf_func,
        args=funcargs,
        threads=len(params),
    )

    rstate0 = np.random.mtrand.RandomState().get_state()
    out = np.zeros((len(params), nsamples))

    for ii in tqdm(range(nsamples // (10 * nwalkers))):  # burn-in
        new_params, new_lnp, new_rstate0 = sampler.run_mcmc(
            all_start_params,
            1,
            log_prob0=start_lnprob0,
            rstate0=rstate0,
            store=False,
        )
        all_start_params = new_params
        start_lnprob0 = new_lnp
        rstate0 = new_rstate0

    for ii in tqdm(range(nsamples // nwalkers + 1)):  # burn-in
        new_params, new_lnp, new_rstate0 = sampler.run_mcmc(
            all_start_params,
            1,
            log_prob0=start_lnprob0,
            rstate0=rstate0,
            store=False,
        )
        all_start_params = new_params
        start_lnprob0 = new_lnp
        rstate0 = new_rstate0
        nleft = nsamples - ii * nwalkers
        if nleft < nwalkers:
            out[:, ii * nwalkers :] = new_params.T[:, :nleft]
        else:
            out[:, ii * nwalkers : (ii + 1) * nwalkers] = new_params.T

    samples = out

    if plots:
        plot_samples(samples, fitc, fitvoro, ro=ro, vo=vo)

    return samples


# /def


# --------------------------------------------------------------------------


def plot_samples(
    samples,
    fitc,
    fitvoro,
    addpal5=False,
    addgd1=False,
    ro=_REFR0,
    vo=_REFV0,
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
        labels.extend(
            [r"$R_0 / \mathrm{kpc}$", r"$V_c(R_0) / \mathrm{km\,s}^{-1}$"]
        )
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
    # if save_figures:
    #     plt.savefig(
    #         os.path.join(
    #             os.getenv("PAPERSDIR"), "2016-mwhalo-shape", "forcefield.pdf"
    #         ),
    #         bbox_inches="tight",
    #     )
    if savefig is not None:
        fig.savefig(savefig)

    return fig


###############################################################################
# END
