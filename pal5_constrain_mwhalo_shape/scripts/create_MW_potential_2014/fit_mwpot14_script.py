# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   :
# AUTHOR  :
# PROJECT :
#
# ----------------------------------------------------------------------------

# Docstring
"""MWPotential2014-varyc.

This notebook contains the fits of sections 2 and 6 of three-component
Milky-Way potential models to a variety of dynamical data and the newly
derived Pal 5 and GD-1 measurements. A variety of fits are explored, most of
which are described in the paper. The full set of two-dimensional PDFs is also
incldued for each fit. Figures 1 and 9 in the paper are produced by this
notebook. Figure 10 of the best-fit force field and the constraints from disk
stars, Pal 5, and GD-1 data is also made by this notebook.

Routing Listings
----------------
make_parser
main

"""

__author__ = "Jo Bovy"
__maintainer__ = "Nathaniel Starkman"

__all__ = [
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
import numpy as np
from tqdm import tqdm

# matplotlib
# fmt: off
import matplotlib; matplotlib.use('Agg')
# fmt: on
import matplotlib.pyplot as plt

# galpy
from galpy.util import (
    bovy_plot,
    save_pickles,
)

from typing import Optional

# PROJECT-SPECIFIC

from ...mw_pot import fit as fit_pot, sample as sample_pot, plot_samples
from . import script_util as su


###############################################################################
# PARAMETERS

np.random.seed(1)  # set random number seed. TODO use numpy1.8 generator

save_figures = False  # TODO needs papers-directory


###############################################################################
# CODE
###############################################################################


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
        the parser with arguments fpath, f

    """
    parser = argparse.ArgumentParser(
        prog="fit_mwpot15_script",
        description="Fit basic MWPotential2014",
        add_help=~inheritable,
        conflict_handler="resolve" if ~inheritable else "error",
    )
    parser.add_argument(
        "-f",
        "--figure",
        default="figures/mwpot14/",
        type=str,
        help="figure save folder",
        dest="fpath",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/",
        type=str,
        help="output save folder",
        dest="opath",
    )

    return parser


# /def


# ------------------------------------------------------------------------


def main(args: Optional[list] = None, opts: Optional[argparse.Namespace] = None):
    """Fit MWPotential2014 Script Function.

    Parameters
    ----------
    args : list, optional
        an optional single argument that holds the sys.argv list,
        except for the script name (e.g., argv[1:])

    """
    if opts is not None and args is None:
        pass
    else:
        parser = make_parser()
        opts = parser.parse_args(args)

    fpath = opts.fpath + "/" if not opts.fpath.endswith("/") else opts.fpath
    opath = opts.opath + "/" if not opts.opath.endswith("/") else opts.opath

    # ----------------------------------------------------------
    # Basic, Bovy (2015) fit with $c=1$

    # -----------------------
    # Using the Clemens CO terminal-velocity data:

    plt.figure(figsize=(16, 5))
    p_b15 = fit_pot(fitc=False, c=1.0, plots=fpath + "Clemens-c_1.pdf")
    plt.close()

    # -----------------------
    # Using the McClure-Griffiths & Dickey HI terminal-velocity data instead

    plt.figure(figsize=(16, 5))
    p_b15_mc16 = fit_pot(fitc=False, c=1.0, mc16=True, plots=fpath + "McClure-c_1.pdf")
    plt.close()

    # We ran the initial analysis with Clemens, which we keep here, until we
    # start interpreting the Pal 5 and GD-1 data; then we switch to the
    # McClure-Griffths & Dickey data. The resulting best-fit parameters here
    # are almost the same, well within each others errors.

    # ----------------------------------------------------------
    # Fits with $c \neq 1$

    # -----------------------

    plt.figure(figsize=(16, 5))
    p_b15_cp5 = fit_pot(fitc=False, c=0.5, plots=fpath + "Clemens-c_0p5.pdf")
    plt.close()

    # -----------------------

    plt.figure(figsize=(16, 5))
    p_b15_c1p5 = fit_pot(fitc=False, c=1.5, plots=fpath + "Clemens-c_1p5.pdf")

    # All look pretty similar...

    # -----------------------

    bf_savefilename = opath + "mwpot14varyc-bf.pkl"
    if os.path.exists(bf_savefilename):
        with open(bf_savefilename, "rb") as savefile:
            cs = pickle.load(savefile)
            bf_params = pickle.load(savefile)
    else:
        cs = np.arange(0.5, 4.1, 0.1)
        bf_params = []
        for c in tqdm(cs):
            dum = fit_pot(fitc=False, c=c, plots=fpath + "mwpot14varyc-bf-fit.pdf",)
            bf_params.append(dum[0])
        save_pickles(bf_savefilename, cs, bf_params)

    # ----------------------------------------------------------
    # Fits with free $c$

    plt.figure(figsize=(16, 5))
    p_b15_cfree = fit_pot(fitc=True, c=None, plots=fpath + "Clemens-c_free.pdf")

    # -----------------------

    samples_savefilename = opath + "mwpot14varyc-samples.pkl"
    if os.path.exists(samples_savefilename):
        with open(samples_savefilename, "rb") as savefile:
            s = pickle.load(savefile)
    else:
        s = sample_pot(
            nsamples=100000,
            params=p_b15_cfree[0],
            fitc=True,
            c=None,
            plots=fpath + "mwpot14varyc-samples.pdf",
            _use_emcee=True,
        )
        save_pickles(samples_savefilename, s)

    # -----------------------

    plt.figure()
    plot_samples(s, True, False, savefig=fpath + "varyc-samples-corner.pdf")

    # -----------------------

    bovy_plot.bovy_print(
        axes_labelsize=17.0,
        text_fontsize=12.0,
        xtick_labelsize=15.0,
        ytick_labelsize=15.0,
    )
    su.plot_mcmc_c(
        s, False, cs, bf_params, savefig=fpath + "varyc-samples-dependence.pdf",
    )

    # -----------------------

    plt.figure(figsize=(4, 4))
    dum = bovy_plot.bovy_hist(
        s[7],
        bins=36,
        histtype="step",
        lw=2.0,
        xlabel=r"$c/a$",
        xrange=[0.0, 4.0],
        normed=True,
    )
    sortedc = np.array(sorted(s[7]))
    plt.title(
        "2.5%% and 0.5%% lower limits: %.2f, %.2f"
        % (
            sortedc[int(np.floor(0.025 * len(sortedc)))],
            sortedc[int(np.floor(0.005 * len(sortedc)))],
        )
    )
    plt.savefig(fpath + "varyc-samples-shape_hist.pdf")

    # -------------------------------------------
    # Also fitting $R_0$ and $V_c(R_0)$

    plt.figure(figsize=(16, 5))
    p_b15_voro = fit_pot(
        fitc=True, c=None, fitvoro=True, plots=fpath + "varyc-fitvoro-samples.pdf",
    )

    # -----------------------

    samples_savefilename = opath + "mwpot14varyc-fitvoro-samples.pkl"
    if os.path.exists(samples_savefilename):
        with open(samples_savefilename, "rb") as savefile:
            s = pickle.load(savefile)
    else:
        s = sample_pot(
            nsamples=100000,
            params=p_b15_voro[0],
            fitc=True,
            c=None,
            plots=fpath + "mwpot14varyc-fitvoro-samples.pdf",
            fitvoro=True,
        )
        save_pickles(samples_savefilename, s)

    # -----------------------

    plt.figure()
    plot_samples(
        s, True, True, savefig=fpath + "varyc-fitvoro-samples-corner.pdf",
    )

    # -----------------------

    plt.figure()
    bovy_plot.bovy_print(
        axes_labelsize=17.0,
        text_fontsize=12.0,
        xtick_labelsize=15.0,
        ytick_labelsize=15.0,
    )
    su.plot_mcmc_c(
        s,
        True,
        cs,
        bf_params,
        add_families=True,
        savefig=fpath + "varyc-fitvoro-samples-dependence.pdf",
    )

    if save_figures:
        plt.savefig(
            os.path.join(
                os.getenv("PAPERSDIR"), "2016-mwhalo-shape", "mwpot14-varyc.pdf",
            ),
            bbox_inches="tight",
        )

    # -----------------------

    plt.figure(figsize=(4, 4))
    dum = bovy_plot.bovy_hist(
        s[9],
        bins=36,
        histtype="step",
        lw=2.0,
        xlabel=r"$c/a$",
        xrange=[0.0, 4.0],
        normed=True,
    )
    sortedc = np.array(sorted(s[9]))
    plt.title(
        "2.5%% and 0.5%% lower limits: %.2f, %.2f"
        % (
            sortedc[int(np.floor(0.025 * len(sortedc)))],
            sortedc[int(np.floor(0.005 * len(sortedc)))],
        )
    )
    plt.savefig(fpath + "varyc-fitvoro-samples-shape_hist.pdf")

    return


# /if


###############################################################################
# END
