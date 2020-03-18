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

__author__ = "Nathaniel Starkman & Jo Bovy"
__maintainer__ = "Nathaniel Starkman"

__all__ = ["main", "make_parser"]


###############################################################################
# IMPORTS

# GENERAL
import os
import copy
import glob
import pickle
import warnings
from argparse import ArgumentParser, Namespace
from types import FunctionType
from typing import Optional

from tqdm import tqdm

import numpy as np
import numba

import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from matplotlib.ticker import NullFormatter
import seaborn as sns
import corner

from galpy.orbit import Orbit
from galpy.util import save_pickles  # switch to astroPHD pickle
from galpy.util import bovy_plot, bovy_coords

# CUSTOM

# PROJECT-SPECIFIC
# fmt: off
import sys; sys.path.insert(0, '../../../')
# fmt: on
from src import MWPotential2014Likelihood
from src import mcmc_util, pal5_util

# fmt: off
from mw_dmhalo_shape_script import _get_pal5varyc, plot_data_add_labels
pal5varyc, cs = _get_pal5varyc()
# fmt: on


###############################################################################
# PARAMETERS

# First read the surface densities
surffile = "../data/mwpot14data/bovyrix13kzdata.csv"
if not surffile is None and os.path.exists(surffile):
    surf = np.loadtxt(surffile, delimiter=",")
    surfrs = surf[:, 2]
    kzs = surf[:, 6]
    kzerrs = surf[:, 7] * 1000.0

# Then the terminal velocities
cl_glon, cl_vterm, cl_corr = MWPotential2014Likelihood.readClemens(dsinl=0.125)
mc_glon, mc_vterm, mc_corr = MWPotential2014Likelihood.readMcClureGriffiths07(
    dsinl=0.125
)
termdata = (cl_glon, cl_vterm, cl_corr, mc_glon, mc_vterm, mc_corr)


_REFR0 = MWPotential2014Likelihood._REFR0
_REFV0 = MWPotential2014Likelihood._REFV0


###############################################################################
# CODE
###############################################################################


@numba.njit()
def evi_harmonic(x):
    # Simple harmonic mean estimator of the evidence, bad!!
    return 1.0 / np.mean(np.exp(-x[:, -1]))


# /def


@numba.njit()
def evi_laplace(x):
    # Laplace estimator of the evidence
    mindx = np.argmax(x[:, -1])
    C = np.cov(x[:, :6], rowvar=False)
    return np.exp(x[mindx, -1]) / np.sqrt(np.linalg.det(C))


# /def


def read_mcmc(
    filename: str = "output/fitsigma/mwpot14-fitsigma-*.dat",
    nburn: Optional[int] = None,
    evi_func: FunctionType = evi_laplace,
    evi_cut: float = -10.0,
    addforces: bool = False,
    addmwpot14weights: bool = False,
    singlepot: Optional[int] = None,
    skip: int = 1,
):
    """Read MCMC

    Parameters
    ----------
    filename : str
    nburn : int, optional
    evi_func : FunctionType, optional
        evidence function
    evi_cut : float, optional
    addforces : bool, optional
    addmwpot14weights : bool, optional
    singlepot : int, optional
        whether to use a single potential, specified by number
        default (None) is to use all
    skip : int, optional


    Returns
    -------
    alldata
    indx : ndarray
        shape (N, )
    weights : ndarray
        shape (N, )
    evis: ndarray
        shape (N, )

    """
    fn = glob.glob(filename)  # filename

    alldata = np.zeros((0, 7 + 2 * addforces))
    indx = np.zeros((0, 1))
    weights = np.zeros((0, 1))
    evis = np.zeros((0, 1))

    for f in tqdm(fn):
        pindx = int(f.split("-")[2].split(".dat")[0])
        # if pindx == 14 or pindx > 27:
        #    print("Remember: skipping 14 and > 27 for now ...")
        #    continue
        if singlepot is not None and not pindx == singlepot:
            continue

        continue_flag = False
        try:
            if nburn is None:
                tnburn = mcmc_util.determine_nburn(f)
            else:
                tnburn = nburn
            tdata = np.loadtxt(f, comments="#", delimiter=",")
            tdata = tdata[tnburn::skip]
            tdata = tdata[tdata[:, -1] > np.nanmax(tdata[:, -1]) + evi_cut]
            if len(tdata) < 100:
                continue_flag = True
        except:
            continue_flag = True  # not enough samples yet

        if continue_flag:
            continue

        # Needs to be before addforces, because evi uses -1 as the lnlike index
        tweights = (
            np.ones((len(tdata), 1)) / float(len(tdata)) * evi_func(tdata)
        )
        evis = np.vstack((evis, np.ones((len(tdata), 1)) * evi_func(tdata)))
        if addforces:
            # Read the potential from the file
            with open(f, "r") as savefile:  # not byte b/c split with str
                line1 = savefile.readline()
            potparams = [float(s) for s in (line1.split(":")[1].split(","))]
            forces = np.empty((len(tdata), 2))
            for ee, c in enumerate(tdata[:, 0]):
                tvo = tdata[ee, 1] * _REFV0
                pot = MWPotential2014Likelihood.setup_potential(
                    potparams, c, False, False, _REFR0, tvo
                )
                forces[ee, :] = MWPotential2014Likelihood.force_pal5(
                    pot, 23.46, _REFR0, tvo
                )[:2]
            tdata = np.hstack((tdata, forces))
        if addmwpot14weights:
            # Not terribly useful right now
            # Add the relative importance weights of this (c,vc) compared to the one
            # that this potential was sampled from
            # Read the potential from the file
            with open(f, "rb") as savefile:
                line1 = savefile.readline()
            potparams = [float(s) for s in (line1.split(":")[1].split(","))]
            # Also load the samples to find the c that this set was sampled with
            with open("mwpot14varyc-samples.pkl", "rb") as savefile:
                s = pickle.load(savefile)
            rndindx = np.argmin(np.fabs(s[0] - potparams[0]))
            pot_params = s[:, rndindx]
            print(pot_params[7])
            base_like = MWPotential2014Likelihood.like_func(
                pot_params,
                pot_params[7],
                surfrs,
                kzs,
                kzerrs,
                termdata,
                700.0,
                False,
                False,
                False,
                False,
                _REFR0,
                220.0,
            )
            for ee, c in enumerate(tdata[:, 0]):
                tvo = tdata[ee, 1] * _REFV0
                tweights[ee] *= np.exp(
                    -MWPotential2014Likelihood.like_func(
                        pot_params,
                        c,
                        surfrs,
                        kzs,
                        kzerrs,
                        termdata,
                        700.0,
                        False,
                        False,
                        False,
                        False,
                        _REFR0,
                        tvo,
                    )  # last one is no vo prior
                    + base_like
                )
        # Only keep
        alldata = np.vstack((alldata, tdata))
        indx = np.vstack(
            (indx, np.zeros((len(tdata), 1), dtype="int") + pindx)
        )
        weights = np.vstack((weights, tweights))

    return alldata, indx[:, 0], weights[:, 0], evis[:, 0]


# /def


def plot_corner(alldata, weights=None, addvcprior=False, addforces=False):
    alldata = copy.deepcopy(alldata)
    weights = copy.deepcopy(weights)
    # First adjust for factors
    alldata[:, 1] *= _REFV0
    if addvcprior:
        weights *= np.exp(-0.5 * (alldata[:, 1] - 220.0) ** 2.0 / 100.0)
    alldata[:, 2] *= 22.0
    alldata[:, 5] = 0.4 * np.exp(alldata[:, 5])
    trange = [
        (0.5, 1.5),
        (200.0, 250.0),
        (19.0, 24.0),
        (-0.4, 0.4),
        (-0.1, 0.1),
        (0.1, 1.0),
    ]
    labels = [
        r"$c$",
        r"$V_c(R_0)$",
        r"$D_{\mathrm{Pal\ 5}}$",
        r"$\mu_\parallel$",
        r"$\mu_\perp$",
        r"$\sigma_v$",
    ]
    if addforces or alldata.shape[1] == 9:  # forces, don't plot likelihood
        alldata = alldata[:, [0, 1, 2, 3, 4, 5, 7, 8]]
        trange.extend([(-1.15, -0.6), (-2.5, -1.25)])
        labels.extend([r"$F_{R,\mathrm{Pal\ 5}}$", r"$F_{Z,\mathrm{Pal\ 5}}$"])
    else:  # no forces, plot likelihood as well
        alldata = alldata[:, :7]
        trange.append((-30.0, -18.0))
        labels.append(r"$\ln \mathcal{L}$")
    corner.corner(
        alldata,
        quantiles=[0.16, 0.5, 0.84],
        range=trange,
        weights=weights,
        labels=labels,
        show_titles=True,
        title_args={"fontsize": 12},
    )


# /def


###############################################################################
# Command Line
###############################################################################


def make_parser(inheritable=False):
    """Make ArgumentParser for fit_mwpot15_script.

    Parameters
    ----------
    inheritable: bool
        whether the parser can be inherited from (default False).
        if True, sets ``add_help=False`` and ``conflict_hander='resolve'``

    Returns
    -------
    parser: ArgumentParser

    """
    parser = ArgumentParser(
        prog="constrain_potential.py with Pal5",
        description="constrain potential from Pal5 data.",
        add_help=~inheritable,
        conflict_handler="resolve" if ~inheritable else "error",
    )
    parser.add_argument(
        "-f",
        "--figure",
        default="figures/",
        type=str,
        help="figure save folder",
        dest="fpath",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/fitsigma/",
        type=str,
        help="output save folder",
        dest="opath",
    )

    return parser


# /def


# ------------------------------------------------------------------------


def main(args: Optional[list] = None, opts: Optional[Namespace] = None):
    """Fit MWPotential2014 Script Function.

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

    fpath = opts.fpath + "/" if not opts.fpath.endswith("/") else opts.fpath
    opath = opts.opath + "/" if not opts.opath.endswith("/") else opts.opath

    # plot chains
    print("Plotting Chains")
    fig = mcmc_util.plot_chains(opath)
    fig.savefig(fpath + "chains.pdf")
    plt.close(fig)

    print("Need to continue chains:", end=" ")
    for i in range(32):
        ngood = mcmc_util.determine_nburn(
            filename=opath + f"mwpot14-fitsigma-{i:02}.dat",
            return_nsamples=True,
        )
        if ngood < 4000:
            print(f"{i:02} (N={ngood})", end=", ")

    ###############################################################
    # RESULTING PDFS
    # --------------
    data, _, weights, _ = read_mcmc(nburn=None, skip=1, evi_func=lambda x: 1.0)

    plot_corner(data, weights=weights)
    plt.savefig("figures/PDFs/corner.pdf")
    plt.close()

    # --------------

    savefilename = "pal5_forces_mcmc.pkl"
    if not os.path.exists(savefilename):
        data_wf, index_wf, weights_wf, evi_wf = read_mcmc(
            nburn=None, addforces=True, skip=1, evi_func=lambda x: 1.0
        )
        save_pickles(savefilename, data_wf, index_wf, weights_wf, evi_wf)
    else:
        with open(savefilename, "rb") as savefile:
            data_wf = pickle.load(savefile)
            index_wf = pickle.load(savefile)
            weights_wf = pickle.load(savefile)
            evi_wf = pickle.load(savefile)

    # --------------
    plot_corner(data_wf, weights=weights_wf, addvcprior=False)
    plt.savefig("figures/PDFs/corner_wf.pdf")
    plt.close()

    # --------------
    # Which potential is preferred?
    data_noforce, potindx, weights, evidences = read_mcmc(
        evi_func=evi_harmonic
    )

    fig = plt.figure(figsize=(6, 4))
    bovy_plot.bovy_plot(
        potindx,
        np.log(evidences),
        "o",
        xrange=[-1, 34],
        yrange=[-35, -22],
        xlabel=r"$\mathrm{Potential\ index}$",
        ylabel=r"$\ln\ \mathrm{evidence}$",
    )
    data_noforce, potindx, weights, evidences = read_mcmc(evi_func=evi_laplace)
    bovy_plot.bovy_plot(potindx, np.log(evidences) - 30.0, "d", overplot=True)
    data_noforce, potindx, weights, evidences = read_mcmc(
        evi_func=lambda x: np.exp(np.amax(x[:, -1]))
    )
    bovy_plot.bovy_plot(potindx, np.log(evidences) - 8.0, "s", overplot=True)
    data_noforce, potindx, weights, evidences = read_mcmc(
        evi_func=lambda x: np.exp(-25.0)
        if (np.log(evi_harmonic(x)) > -25.0)
        else np.exp(-50.0)
    )
    bovy_plot.bovy_plot(potindx, np.log(evidences), "o", overplot=True)
    plt.savefig("figures/PDFs/preferred_pot.pdf")
    plt.close()

    ###############################################################
    # Look at the results for individual potentials

    # --------------
    # The flattening $c$
    npot = 32
    nwalkers = 12

    plt.figure(figsize=(16, 6))
    cmap = cm.plasma
    maxl = np.zeros((npot, 2))
    for en, ii in enumerate(range(npot)):
        data_ip, _, weights_ip, evi_ip = read_mcmc(
            singlepot=ii, evi_func=evi_harmonic
        )
        try:
            maxl[en, 0] = np.amax(data_ip[:, -1])
            maxl[en, 1] = np.log(evi_ip[0])
        except ValueError:
            maxl[en] = -10000000.0
        plt.subplot(2, 4, en // 4 + 1)
        bovy_plot.bovy_hist(
            data_ip[:, 0],
            range=[0.5, 2.0],
            bins=26,
            histtype="step",
            color=cmap((en % 4) / 3.0),
            normed=True,
            xlabel=r"$c$",
            lw=1.5,
            overplot=True,
        )
        if en % 4 == 0:
            bovy_plot.bovy_text(
                r"$\mathrm{Potential\ %i\ to\ % i}$" % (en, en + 3),
                size=17.0,
                top_left=True,
            )
    plt.tight_layout()
    plt.savefig("figures/flattening_c.pdf")
    plt.close()

    ###############################################################
    # What is the effective prior in $(F_R,F_Z)$?
    frfzprior_savefilename = "frfzprior.pkl"
    if not os.path.exists(frfzprior_savefilename):
        # Compute for each potential separately
        nvoc = 10000
        ro = 8.0
        npot = 32
        fs = np.zeros((2, nvoc, npot))
        for en, ii in tqdm(enumerate(range(npot))):
            fn = f"output/fitsigma/mwpot14-fitsigma-{i:02}.dat"
            # Read the potential parameters
            with open(fn, "rb") as savefile:
                line1 = savefile.readline()
            potparams = [float(s) for s in (line1.split(":")[1].split(","))]
            for jj in range(nvoc):
                c = np.random.uniform() * 1.5 + 0.5
                tvo = np.random.uniform() * 50.0 + 200.0
                pot = MWPotential2014Likelihood.setup_potential(
                    potparams, c, False, False, ro, tvo
                )
                fs[:, jj, ii] = np.array(
                    MWPotential2014Likelihood.force_pal5(pot, 23.46, ro, tvo)
                )[:2]
        save_pickles(frfzprior_savefilename, fs)
    else:
        with open(frfzprior_savefilename, "rb") as savefile:
            fs = pickle.load(savefile)

    # --------------
    plt.figure(figsize=(6, 6))
    bovy_plot.scatterplot(
        fs[0].flatten(),
        fs[1].flatten(),
        "k,",
        xrange=[-1.75, -0.25],
        yrange=[-2.5, -1.2],
        xlabel=r"$F_R(\mathrm{Pal\ 5})$",
        ylabel=r"$F_Z(\mathrm{Pal\ 5})$",
        onedhists=True,
    )
    bovy_plot.scatterplot(
        data_wf[:, 7],
        data_wf[:, 8],
        weights=weights_wf,
        bins=26,
        xrange=[-1.75, -0.25],
        yrange=[-2.5, -1.2],
        justcontours=True,
        cntrcolors="w",
        overplot=True,
        onedhists=True,
    )
    plt.axvline(-0.81, color=sns.color_palette()[0])
    plt.axhline(-1.85, color=sns.color_palette()[0])
    plt.savefig("figures/effective_force_prior.pdf")
    plt.close()

    # --------------
    # The ratio of the posterior and the prior

    bovy_plot.bovy_print(
        axes_labelsize=17.0,
        text_fontsize=12.0,
        xtick_labelsize=14.0,
        ytick_labelsize=14.0,
    )
    plt.figure(figsize=(12.5, 4))

    def axes_white():
        for k, spine in plt.gca().spines.items():  # ax.spines is a dictionary
            spine.set_color("w")
        plt.gca().tick_params(axis="x", which="both", colors="w")
        plt.gca().tick_params(axis="y", which="both", colors="w")
        [t.set_color("k") for t in plt.gca().xaxis.get_ticklabels()]
        [t.set_color("k") for t in plt.gca().yaxis.get_ticklabels()]
        return None

    bins = 32
    trange = [[-1.75, -0.25], [-2.5, -1.2]]
    tw = copy.deepcopy(weights_wf)
    tw[index_wf == 14] = 0.0  # Didn't converge properly
    H_prior, xedges, yedges = np.histogram2d(
        fs[0].flatten(), fs[1].flatten(), bins=bins, range=trange, normed=True
    )
    H_post, xedges, yedges = np.histogram2d(
        data_wf[:, 7],
        data_wf[:, 8],
        weights=tw,
        bins=bins,
        range=trange,
        normed=True,
    )
    H_like = H_post / H_prior
    H_like[H_prior == 0.0] = 0.0
    plt.subplot(1, 3, 1)
    bovy_plot.bovy_dens2d(
        H_prior.T,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        xrange=[xedges[0], xedges[-1]],
        yrange=[yedges[0], yedges[-1]],
        xlabel=r"$F_R(\mathrm{Pal\ 5})\,(\mathrm{km\,s}^{-1}\,\mathrm{Myr}^{-1})$",
        ylabel=r"$F_Z(\mathrm{Pal\ 5})\,(\mathrm{km\,s}^{-1}\,\mathrm{Myr}^{-1})$",
        gcf=True,
    )
    bovy_plot.bovy_text(
        r"$\mathbf{Prior}$", top_left=True, size=19.0, color="w"
    )
    axes_white()
    plt.subplot(1, 3, 2)
    bovy_plot.bovy_dens2d(
        H_post.T,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        xrange=[xedges[0], xedges[-1]],
        yrange=[yedges[0], yedges[-1]],
        xlabel=r"$F_R(\mathrm{Pal\ 5})\,(\mathrm{km\,s}^{-1}\,\mathrm{Myr}^{-1})$",
        gcf=True,
    )
    bovy_plot.bovy_text(
        r"$\mathbf{Posterior}$", top_left=True, size=19.0, color="w"
    )
    axes_white()
    plt.subplot(1, 3, 3)
    bovy_plot.bovy_dens2d(
        H_like.T,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        vmin=0.1,
        vmax=4.0,
        xrange=[xedges[0], xedges[-1]],
        yrange=[yedges[0], yedges[-1]],
        xlabel=r"$F_R(\mathrm{Pal\ 5})\,(\mathrm{km\,s}^{-1}\,\mathrm{Myr}^{-1})$",
        gcf=True,
    )
    bovy_plot.bovy_text(
        r"$\mathbf{Likelihood}$", top_left=True, size=19.0, color="w"
    )
    axes_white()

    def qline(FR, q=0.95):
        return 2.0 * FR / q ** 2.0

    q = 0.94
    plt.plot([-1.25, -0.2], [qline(-1.25, q=q), qline(-0.2, q=q)], "w--")
    bovy_plot.bovy_text(-1.7, -2.2, r"$q_\Phi = 0.94$", size=16.0, color="w")
    plt.plot((-1.25, -1.02), (-2.19, qline(-1.02, q=q)), "w-", lw=0.8)

    plt.tight_layout()
    plt.savefig(
        "figures/2016-mwhalo-shape/pal5post.pdf", bbox_inches="tight",
    )
    plt.close()

    # --------------
    # Projection onto the direction perpendicular to constant $q = 0.94$:
    frs = np.tile(0.5 * (xedges[:-1] + xedges[1:]), (len(yedges) - 1, 1)).T
    fzs = np.tile(0.5 * (yedges[:-1] + yedges[1:]), (len(xedges) - 1, 1))
    plt.figure(figsize=(6, 4))
    txlabel = r"$F_\perp$"
    dum = bovy_plot.bovy_hist(
        (-2.0 * (frs + 0.8) + 0.94 ** 2.0 * (fzs + 1.82)).flatten(),
        weights=H_prior.flatten(),
        bins=21,
        histtype="step",
        lw=2.0,
        xrange=[-1.5, 1.5],
        xlabel=txlabel,
        normed=True,
    )
    dum = bovy_plot.bovy_hist(
        (-2.0 * (frs + 0.8) + 0.94 ** 2.0 * (fzs + 1.82)).flatten(),
        weights=H_post.flatten(),
        bins=21,
        histtype="step",
        lw=2.0,
        overplot=True,
        xrange=[-1.5, 1.5],
        normed=True,
    )
    dum = bovy_plot.bovy_hist(
        (-2.0 * (frs + 0.8) + 0.94 ** 2.0 * (fzs + 1.82)).flatten(),
        weights=H_like.flatten(),
        bins=21,
        histtype="step",
        lw=2.0,
        overplot=True,
        xrange=[-1.5, 1.5],
        normed=True,
    )
    plt.savefig("figures/PDFs/projection_perp_to_q0p94.pdf")
    plt.close()

    mq = (
        np.sum(
            (-2.0 * (frs + 0.8) + 0.94 ** 2.0 * (fzs + 1.82)).flatten()
            * H_like.flatten()
        )
    ) / np.sum(H_like.flatten())

    print(
        mq,
        np.sqrt(
            (
                np.sum(
                    (
                        (
                            -2.0 * (frs + 0.8) + 0.94 ** 2.0 * (fzs + 1.82)
                        ).flatten()
                        - mq
                    )
                    ** 2.0
                    * H_like.flatten()
                )
            )
            / np.sum(H_like.flatten())
        ),
    )

    # --------------
    # Projection onto the direction parallel to constant $q = 0.94$:
    frs = np.tile(0.5 * (xedges[:-1] + xedges[1:]), (len(yedges) - 1, 1)).T
    fzs = np.tile(0.5 * (yedges[:-1] + yedges[1:]), (len(xedges) - 1, 1))
    plt.figure(figsize=(6, 4))
    txlabel = r"$F_\parallel$"
    dum = bovy_plot.bovy_hist(
        (0.94 ** 2.0 * (frs + 0.8) + 2.0 * (fzs + 1.82)).flatten(),
        weights=H_prior.flatten(),
        bins=21,
        histtype="step",
        lw=2.0,
        xrange=[-2.5, 2.5],
        xlabel=txlabel,
        normed=True,
    )
    dum = bovy_plot.bovy_hist(
        (0.94 ** 2.0 * (frs + 0.8) + 2.0 * (fzs + 1.82)).flatten(),
        weights=H_post.flatten(),
        bins=21,
        histtype="step",
        lw=2.0,
        overplot=True,
        xrange=[-2.5, 2.5],
        normed=True,
    )
    dum = bovy_plot.bovy_hist(
        (0.94 ** 2.0 * (frs + 0.8) + 2.0 * (fzs + 1.82)).flatten(),
        weights=H_like.flatten(),
        bins=21,
        histtype="step",
        lw=2.0,
        overplot=True,
        xrange=[-2.5, 2.5],
        normed=True,
    )
    plt.savefig("figures/PDFs/projection_prll_to_q0p94.pdf")
    plt.close()
    mq = (
        np.sum(
            (0.94 ** 2.0 * (frs + 0.8) + 2.0 * (fzs + 1.82)).flatten()
            * H_like.flatten()
        )
    ) / np.sum(H_like.flatten())
    print(
        mq,
        np.sqrt(
            (
                np.sum(
                    (
                        (
                            0.94 ** 2.0 * (frs + 0.8) + 2.0 * (fzs + 1.82)
                        ).flatten()
                        - mq
                    )
                    ** 2.0
                    * H_like.flatten()
                )
            )
            / np.sum(H_like.flatten())
        ),
    )

    # --------------
    # Thus, there is only a weak constraint on $F_\parallel$.

    nrow = int(np.ceil(npot / 4.0))
    plt.figure(figsize=(16, nrow * 4))
    for en, ii in enumerate(range(npot)):
        plt.subplot(nrow, 4, en + 1)
        if ii % 4 == 0:
            tylabel = r"$F_Z(\mathrm{Pal\ 5})$"
        else:
            tylabel = None
        if ii // 4 == nrow - 1:
            txlabel = r"$F_R(\mathrm{Pal\ 5})$"
        else:
            txlabel = None
        bovy_plot.scatterplot(
            fs[0][:, en],
            fs[1][:, en],
            "k,",
            xrange=[-1.75, -0.25],
            yrange=[-2.5, -1.0],
            xlabel=txlabel,
            ylabel=tylabel,
            gcf=True,
        )
        bovy_plot.scatterplot(
            data_wf[:, 7],
            data_wf[:, 8],
            weights=weights_wf,
            bins=26,
            xrange=[-1.75, -0.25],
            yrange=[-2.5, -1.0],
            justcontours=True,
            cntrcolors="w",
            overplot=True,
        )
        bovy_plot.scatterplot(
            data_wf[index_wf == ii, 7],
            data_wf[index_wf == ii, 8],
            weights=weights_wf[index_wf == ii],
            bins=26,
            xrange=[-1.75, -0.25],
            yrange=[-2.5, -1.0],
            justcontours=True,
            cntrcolors=sns.color_palette()[2],
            overplot=True,
        )
        plt.axvline(-0.80, color=sns.color_palette()[0])
        plt.axhline(-1.83, color=sns.color_palette()[0])
        bovy_plot.bovy_text(
            r"$\mathrm{Potential}\ %i$" % ii, size=17.0, top_left=True
        )
    plt.savefig("figures/PDFs/constain_F_prll.pdf")
    plt.close()

    # --------------
    # Let's plot four representative ones for the paper:
    bovy_plot.bovy_print(
        axes_labelsize=17.0,
        text_fontsize=12.0,
        xtick_labelsize=14.0,
        ytick_labelsize=14.0,
    )
    nullfmt = NullFormatter()
    nrow = 1

    plt.figure(figsize=(15, nrow * 4))
    for en, ii in enumerate([0, 15, 24, 25]):
        plt.subplot(nrow, 4, en + 1)
        if en % 4 == 0:
            tylabel = r"$F_Z(\mathrm{Pal\ 5})\,(\mathrm{km\,s}^{-1}\,\mathrm{Myr}^{-1})$"
        else:
            tylabel = None
        if en // 4 == nrow - 1:
            txlabel = r"$F_R(\mathrm{Pal\ 5})\,(\mathrm{km\,s}^{-1}\,\mathrm{Myr}^{-1})$"
        else:
            txlabel = None
        bovy_plot.scatterplot(
            fs[0][:, ii],
            fs[1][:, ii],
            "k,",
            bins=31,
            xrange=[-1.75, -0.25],
            yrange=[-2.5, -1.0],
            xlabel=txlabel,
            ylabel=tylabel,
            gcf=True,
        )
        bovy_plot.scatterplot(
            data_wf[:, 7],
            data_wf[:, 8],
            weights=weights_wf,
            bins=21,
            xrange=[-1.75, -0.25],
            yrange=[-2.5, -1.0],
            justcontours=True,
            cntrcolors=sns.color_palette("colorblind")[2],
            cntrls="--",
            cntrlw=2.0,
            overplot=True,
        )
        bovy_plot.scatterplot(
            data_wf[index_wf == ii, 7],
            data_wf[index_wf == ii, 8],
            weights=weights_wf[index_wf == ii],
            bins=21,
            xrange=[-1.75, -0.25],
            yrange=[-2.5, -1.0],
            justcontours=True,
            cntrcolors=sns.color_palette("colorblind")[0],
            cntrlw=2.5,
            overplot=True,
        )
        if en > 0:
            plt.gca().yaxis.set_major_formatter(nullfmt)

    plt.tight_layout()
    plt.savefig(
        "figures/pal5post_examples.pdf", bbox_inches="tight",
    )
    plt.close()

    ###############################################################
    # What about $q_\Phi$?

    bins = 47
    plt.figure(figsize=(6, 4))
    dum = bovy_plot.bovy_hist(
        np.sqrt(2.0 * fs[0].flatten() / fs[1].flatten()),
        histtype="step",
        lw=2.0,
        bins=bins,
        xlabel=r"$q_\mathrm{\Phi}$",
        xrange=[0.7, 1.25],
        normed=True,
    )
    dum = bovy_plot.bovy_hist(
        np.sqrt(16.8 / 8.4 * data_wf[:, -2] / data_wf[:, -1]),
        weights=weights_wf,
        histtype="step",
        lw=2.0,
        bins=bins,
        overplot=True,
        xrange=[0.7, 1.25],
        normed=True,
    )
    mq = np.sum(
        np.sqrt(16.8 / 8.4 * data_wf[:, -2] / data_wf[:, -1]) * weights_wf
    ) / np.sum(weights_wf)
    sq = np.sqrt(
        np.sum(
            (np.sqrt(16.8 / 8.4 * data_wf[:, -2] / data_wf[:, -1]) - mq) ** 2.0
            * weights_wf
        )
        / np.sum(weights_wf)
    )
    print("From posterior samples: q = %.3f +/- %.3f" % (mq, sq))
    Hq_post, xedges = np.histogram(
        np.sqrt(16.8 / 8.4 * data_wf[:, -2] / data_wf[:, -1]),
        weights=weights_wf,
        bins=bins,
        range=[0.7, 1.25],
        normed=True,
    )
    Hq_prior, xedges = np.histogram(
        np.sqrt(2.0 * fs[0].flatten() / fs[1].flatten()),
        bins=bins,
        range=[0.7, 1.25],
        normed=True,
    )
    qs = 0.5 * (xedges[:-1] + xedges[1:])
    Hq_like = Hq_post / Hq_prior
    Hq_like[Hq_post == 0.0] = 0.0
    mq = np.sum(qs * Hq_like) / np.sum(Hq_like)
    sq = np.sqrt(np.sum((qs - mq) ** 2.0 * Hq_like) / np.sum(Hq_like))
    print("From likelihood of samples: q = %.3f +/- %.3f" % (mq, sq))

    plt.savefig("figures/q_phi.pdf")
    plt.close()

    # It appears that $q_\Phi$ is the quantity that is the most strongly constrained by the Pal 5 data.

    ###############################################################
    # A sampling of tracks from the MCMC

    savefilename = "mwpot14-pal5-mcmcTracks.pkl"
    pmdecpar = 2.257 / 2.296
    pmdecperp = -2.296 / 2.257
    if os.path.exists(savefilename):
        with open(savefilename, "rb") as savefile:
            pal5_track_samples = pickle.load(savefile)
            forces = pickle.load(savefile)
            all_potparams = pickle.load(savefile)
            all_params = pickle.load(savefile)
    else:
        np.random.seed(1)
        ntracks = 21
        multi = 8
        pal5_track_samples = np.zeros((ntracks, 2, 6, pal5varyc[0].shape[1]))
        forces = np.zeros((ntracks, 2))
        all_potparams = np.zeros((ntracks, 5))
        all_params = np.zeros((ntracks, 7))
        for ii in range(ntracks):
            # Pick a random potential from among the set, but leave 14 out
            pindx = 14
            while pindx == 14:
                pindx = np.random.permutation(32)[0]
            # Load this potential
            fn = "../pal5_mcmc/mwpot14-fitsigma-%i.dat" % pindx
            with open(fn, "rb") as savefile:
                line1 = savefile.readline()
            potparams = [float(s) for s in (line1.split(":")[1].split(","))]
            all_potparams[ii] = potparams
            # Now pick a random sample from this MCMC chain
            tnburn = mcmc_util.determine_nburn(fn)
            tdata = np.loadtxt(fn, comments="#", delimiter=",")
            tdata = tdata[tnburn::]
            tdata = tdata[np.random.permutation(len(tdata))[0]]
            all_params[ii] = tdata
            tvo = tdata[1] * _REFV0
            pot = MWPotential2014Likelihood.setup_potential(
                potparams, tdata[0], False, False, _REFR0, tvo
            )
            forces[ii, :] = MWPotential2014Likelihood.force_pal5(
                pot, 23.46, ro, tvo
            )[:2]
            # Now compute the stream model for this setup
            dist = tdata[2] * 22.0
            pmra = -2.296 + tdata[3] + tdata[4]
            pmdecpar = 2.257 / 2.296
            pmdecperp = -2.296 / 2.257
            pmdec = -2.257 + tdata[3] * pmdecpar + tdata[4] * pmdecperp
            vlos = -58.7
            sigv = 0.4 * np.exp(tdata[5])
            prog = Orbit(
                [229.018, -0.124, dist, pmra, pmdec, vlos],
                radec=True,
                ro=ro,
                vo=tvo,
                solarmotion=[-11.1, 24.0, 7.25],
            )
            tsdf_trailing, tsdf_leading = pal5_util.setup_sdf(
                pot,
                prog,
                sigv,
                10.0,
                ro,
                tvo,
                multi=multi,
                nTrackChunks=8,
                trailing_only=False,
                verbose=True,
                useTM=False,
            )
            # Compute the track
            for jj, sdf in enumerate([tsdf_trailing, tsdf_leading]):
                trackRADec = bovy_coords.lb_to_radec(
                    sdf._interpolatedObsTrackLB[:, 0],
                    sdf._interpolatedObsTrackLB[:, 1],
                    degree=True,
                )
                trackpmRADec = bovy_coords.pmllpmbb_to_pmrapmdec(
                    sdf._interpolatedObsTrackLB[:, 4],
                    sdf._interpolatedObsTrackLB[:, 5],
                    sdf._interpolatedObsTrackLB[:, 0],
                    sdf._interpolatedObsTrackLB[:, 1],
                    degree=True,
                )
                # Store the track
                pal5_track_samples[ii, jj, 0] = trackRADec[:, 0]
                pal5_track_samples[ii, jj, 1] = trackRADec[:, 1]
                pal5_track_samples[ii, jj, 2] = sdf._interpolatedObsTrackLB[
                    :, 2
                ]
                pal5_track_samples[ii, jj, 3] = sdf._interpolatedObsTrackLB[
                    :, 3
                ]
                pal5_track_samples[ii, jj, 4] = trackpmRADec[:, 0]
                pal5_track_samples[ii, jj, 5] = trackpmRADec[:, 1]
        save_pickles(
            savefilename, pal5_track_samples, forces, all_potparams, all_params
        )

    # --------------
    bovy_plot.bovy_print(
        axes_labelsize=17.0,
        text_fontsize=12.0,
        xtick_labelsize=14.0,
        ytick_labelsize=14.0,
    )
    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, wspace=0.225, hspace=0.1, right=0.94)
    ntracks = pal5_track_samples.shape[0]
    cmap = cm.plasma
    alpha = 0.7
    for ii in range(ntracks):
        tc = cmap((forces[ii, 1] + 2.5) / 1.0)
        for jj in range(2):
            # RA, Dec
            plt.subplot(gs[0])
            plt.plot(
                pal5_track_samples[ii, jj, 0],
                pal5_track_samples[ii, jj, 1],
                "-",
                color=tc,
                alpha=alpha,
            )
            # RA, Vlos
            plt.subplot(gs[1])
            plt.plot(
                pal5_track_samples[ii, jj, 0],
                pal5_track_samples[ii, jj, 3],
                "-",
                color=tc,
                alpha=alpha,
            )
            # RA, Dist
            plt.subplot(gs[2])
            plt.plot(
                pal5_track_samples[ii, jj, 0],
                pal5_track_samples[ii, jj, 2] - all_params[ii, 2] * 22.0,
                "-",
                color=tc,
                alpha=alpha,
            )
            # RA, pm_parallel
            plt.subplot(gs[3])
            plt.plot(
                pal5_track_samples[ii, jj, 0, : 500 + 500 * (1 - jj)],
                np.sqrt(1.0 + (2.257 / 2.296) ** 2.0)
                * (
                    (
                        pal5_track_samples[ii, jj, 4, : 500 + 500 * (1 - jj)]
                        + 2.296
                    )
                    * pmdecperp
                    - (
                        pal5_track_samples[ii, jj, 5, : 500 + 500 * (1 - jj)]
                        + 2.257
                    )
                )
                / (pmdecpar - pmdecperp),
                "-",
                color=tc,
                alpha=alpha,
            )
    plot_data_add_labels(
        p1=(gs[0],), p2=(gs[1],), noxlabel_dec=True, noxlabel_vlos=True
    )
    plt.subplot(gs[2])
    plt.xlim(250.0, 221.0)
    plt.ylim(-3.0, 1.5)
    bovy_plot._add_ticks()
    plt.xlabel(r"$\mathrm{RA}\,(\mathrm{degree})$")
    plt.ylabel(r"$\Delta\mathrm{Distance}\,(\mathrm{kpc})$")
    plt.subplot(gs[3])
    plt.xlim(250.0, 221.0)
    plt.ylim(-0.5, 0.5)
    bovy_plot._add_ticks()
    plt.xlabel(r"$\mathrm{RA}\,(\mathrm{degree})$")
    plt.ylabel(r"$\Delta\mu_\parallel\,(\mathrm{mas\,yr}^{-1})$")
    # Colorbar
    gs2 = gridspec.GridSpec(2, 1, wspace=0.0, left=0.95, right=0.975)
    plt.subplot(gs2[:, -1])
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=-2.5, vmax=-1.5)
    )
    sm._A = []
    CB1 = plt.colorbar(sm, orientation="vertical", cax=plt.gca())
    CB1.set_label(
        r"$F_Z(\mathrm{Pal\ 5})\,(\mathrm{km\,s}^{-1}\,\mathrm{Myr}^{-1})$",
        fontsize=18.0,
    )
    plt.savefig(
        "figures/pal5tracksamples.pdf", bbox_inches="tight",
    )
    plt.close()

    ###############################################################
    # What is the effective prior in $(F_R,F_Z)$?

    from galpy import potential
    from astropy import units as u

    def setup_potential_kuepper(Mhalo, a):
        r"""Mhalo: mass/10^12 Msun
        a: scale length in kpc,
        
        we'll take q_\Phi into account when we evaluate the potential..."""
        return [
            potential.HernquistPotential(
                amp=3.4 * 10.0 ** 10.0 * u.Msun, a=0.7 * u.kpc
            ),
            potential.MiyamotoNagaiPotential(
                amp=10.0 ** 11.0 * u.Msun, a=6.5 * u.kpc, b=0.26 * u.kpc
            ),
            potential.NFWPotential(
                amp=Mhalo * 10.0 ** 12.0 * u.Msun, a=a * u.kpc
            ),
        ]

    def force_pal5_kuepper(pot, qNFW):
        FR = (
            pot[0].Rforce(8.4 * u.kpc, 16.8 * u.kpc)
            + pot[1].Rforce(8.4 * u.kpc, 16.8 * u.kpc)
            + pot[2].Rforce(8.4 * u.kpc, 16.8 * u.kpc / qNFW)
        )
        FZ = (
            pot[0].zforce(8.4 * u.kpc, 16.8 * u.kpc)
            + pot[1].zforce(8.4 * u.kpc, 16.8 * u.kpc)
            + pot[2].zforce(8.4 * u.kpc, 16.8 * u.kpc / qNFW)
        )
        return (FR, FZ)

    # Can we recover the Kuepper et al. (2015) result as prior + $q_\Phi =
    # 0.94 \pm 0.05$ + $V_c(R_0)$ between 200 and 280 km/s? For simplicity
    # we will not vary $R_0$, which should not have a big impact.

    import bovy_mcmc
    import corner

    def kuepper_flattening_post(params, qmean, qerr):
        """A Kuepper et al.-like posterior that consists solely of the priors and a constraint on q_\Phi"""
        Mhalo = params[0]
        a = params[1]
        qNFW = params[2]
        if Mhalo < 0.001 or Mhalo > 10.0:
            return -1000000000000000000.0
        elif a < 0.1 or a > 100.0:
            return -1000000000000000000.0
        elif qNFW < 0.2 or qNFW > 1.8:
            return -1000000000000000000.0
        pot = setup_potential_kuepper(Mhalo, a)
        vcpred = potential.vcirc(pot, 8.0 * u.kpc)
        if vcpred < 200.0 or vcpred > 280.0:
            return -1000000000000000000.0
        FR, FZ = force_pal5_kuepper(pot, qNFW)
        qpred = np.sqrt(2.0 * FR / FZ)
        return -0.5 * (qpred - qmean) ** 2.0 / qerr ** 2.0

    def kuepper_flatteningforce_post(params, qmean, qerr, frfz, frfzerr):
        """A Kuepper et al.-like posterior that consists solely of the priors, a constraint on q_\Phi, and a constraint
        on FR+FZ"""
        Mhalo = params[0]
        a = params[1]
        qNFW = params[2]
        if Mhalo < 0.001 or Mhalo > 10.0:
            return -1000000000000000000.0
        elif a < 0.1 or a > 100.0:
            return -1000000000000000000.0
        elif qNFW < 0.2 or qNFW > 1.8:
            return -1000000000000000000.0
        pot = setup_potential_kuepper(Mhalo, a)
        vcpred = potential.vcirc(pot, 8.0 * u.kpc)
        if vcpred < 200.0 or vcpred > 280.0:
            return -1000000000000000000.0
        FR, FZ = force_pal5_kuepper(pot, qNFW)
        qpred = np.sqrt(2.0 * FR / FZ)
        frfzpred = (FR + 0.8) + (FZ + 1.83)
        return (
            -0.5 * (qpred - qmean) ** 2.0 / qerr ** 2.0
            - 0.5 * (frfz - frfzpred) ** 2.0 / frfzerr ** 2.0
        )

    def sample_kuepper_flattening_post(nsamples, qmean, qerr):
        params = [1.58, 37.9, 0.95]
        funcargs = (qmean, qerr)
        samples = bovy_mcmc.markovpy(
            params,
            0.2,
            lambda x: kuepper_flattening_post(x, *funcargs),
            (),
            isDomainFinite=[[False, False] for ii in range(len(params))],
            domain=[[0.0, 0.0] for ii in range(len(params))],
            nsamples=nsamples,
            nwalkers=2 * len(params),
        )
        samples = np.array(samples).T
        return samples

    def sample_kuepper_flatteningforce_post(
        nsamples, qmean, qerr, frfz, frfzerr
    ):
        params = [1.58, 37.9, 0.95]
        funcargs = (qmean, qerr, frfz, frfzerr)
        samples = bovy_mcmc.markovpy(
            params,
            0.2,
            lambda x: kuepper_flatteningforce_post(x, *funcargs),
            (),
            isDomainFinite=[[False, False] for ii in range(len(params))],
            domain=[[0.0, 0.0] for ii in range(len(params))],
            nsamples=nsamples,
            nwalkers=2 * len(params),
        )
        samples = np.array(samples).T
        return samples

    def plot_kuepper_samples(samples):
        labels = [
            r"$M_{\mathrm{halo}} / (10^{12}\,M_\odot)$",
            r"$q_z$",
            r"$a / \mathrm{kpc}$",
        ]
        ranges = [(0.0, 4.0), (0.2, 1.8), (0.0, 100.0)]
        corner.corner(
            samples[[0, 2, 1]].T,
            quantiles=[0.16, 0.5, 0.84],
            labels=labels,
            show_titles=True,
            title_args={"fontsize": 12},
            range=ranges,
        )

    s = sample_kuepper_flattening_post(50000, 0.94, 0.05)
    plot_kuepper_samples(s)
    plt.savefig("figures/kuepper_samples.pdf")
    plt.close()

    # The constraint on the potential flattening gets you far, but there
    # is more going on (the constraint on the halo mass and scale
    # parameter appear to come completely from the $V_c(R_0)$ constraint).
    # Let's add the weak constraint on the sum of the forces, scaled to
    # Kuepper et al.'s best-fit acceleration (which is higher than ours):

    s = sample_kuepper_flatteningforce_post(50000, 0.94, 0.05, -0.83, 0.36)
    plot_kuepper_samples(s)
    plt.savefig("figures/kuepper_samples_flatF.pdf")
    plt.close()

    # This gets a tight relation between $M_\mathrm{halo}$ and the scale
    # parameter of the halo, but does not lead to a constraint on either
    # independently; the halo potential flattening constraint is that of
    # Kuepper et al. Based on this, it appears that the information that
    # ties down $M_\mathrm{halo}$ comes from the overdensities, which may
    # be spurious (Thomas et al. 2016) and whose use in dynamical modeling
    # is dubious anyway.

    # What is Kuepper et al.'s prior on $a_{\mathrm{Pal\ 5}}$?

    apal5s = []
    ns = 100000
    for ii in range(ns):
        Mh = np.random.uniform() * (10.0 - 0.001) + 0.001
        a = np.random.uniform() * (100.0 - 0.1) + 0.1
        q = np.random.uniform() * (1.8 - 0.2) + 0.2
        pot = setup_potential_kuepper(Mh, a)
        FR, FZ = force_pal5_kuepper(pot, q)
        apal5s.append(np.sqrt(FR ** 2.0 + FZ ** 2.0))
    apal5s = np.array(apal5s)

    plt.figure(figsize=(6, 4))
    dum = bovy_plot.bovy_hist(
        apal5s,
        range=[0.0, 10.0],
        lw=2.0,
        bins=51,
        histtype="step",
        normed=True,
    )
    plt.savefig("figures/kuepper_samples_prior.pdf")
    plt.close()

    # This prior is clearly non-flat, and has a very long tail toward higher
    # accelerations that may be biasing their inferred acceleration high.


# /def


# ------------------------------------------------------------------------

if __name__ == "__main__":

    main(args=None, opts=None)

# /if


###############################################################################
# END
