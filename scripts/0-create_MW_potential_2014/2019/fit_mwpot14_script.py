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
import numpy
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

# PROJECT-SPECIFIC
import script_util as su


###############################################################################
# PARAMETERS

numpy.random.seed(1)  # set random number seed. TODO use numpy1.8 generator

save_figures = False  # TODO needs papers-directory


###############################################################################
# Command Line
###############################################################################


if __name__ == "__main__":

    # ----------------------------------------------------------
    # Basic, Bovy (2015) fit with $c=1$

    # -----------------------
    # Using the Clemens CO terminal-velocity data:

    plt.figure(figsize=(16, 5))
    p_b15 = su.fit(fitc=False, c=1.0, plots="figures/mwpot14/Clemens-c_1.pdf")

    # -----------------------
    # Using the McClure-Griffiths & Dickey HI terminal-velocity data instead

    plt.figure(figsize=(16, 5))
    p_b15_mc16 = su.fit(
        fitc=False, c=1.0, mc16=True, plots="figures/mwpot14/McClure-c_1.pdf"
    )

    # We ran the initial analysis with Clemens, which we keep here, until we
    # start interpreting the Pal 5 and GD-1 data; then we switch to the
    # McClure-Griffths & Dickey data. The resulting best-fit parameters here
    # are almost the same, well within each others errors.

    # ----------------------------------------------------------
    # Fits with $c \neq 1$

    # -----------------------

    plt.figure(figsize=(16, 5))
    p_b15_cp5 = su.fit(
        fitc=False, c=0.5, plots="figures/mwpot14/Clemens-c_0p5.pdf"
    )

    # -----------------------

    plt.figure(figsize=(16, 5))
    p_b15_c1p5 = su.fit(
        fitc=False, c=1.5, plots="figures/mwpot14/Clemens-c_1p5.pdf"
    )

    # All look pretty similar...

    # -----------------------

    bf_savefilename = "output/mwpot14varyc-bf.pkl"
    if os.path.exists(bf_savefilename):
        with open(bf_savefilename, "rb") as savefile:
            cs = pickle.load(savefile)
            bf_params = pickle.load(savefile)
    else:
        cs = numpy.arange(0.5, 4.1, 0.1)
        bf_params = []
        for c in tqdm(cs):
            dum = su.fit(fitc=False, c=c)
            bf_params.append(dum[0])
        save_pickles(bf_savefilename, cs, bf_params)

    # ----------------------------------------------------------
    # Fits with free $c$

    plt.figure(figsize=(16, 5))
    p_b15_cfree = su.fit(
        fitc=True, c=None, plots="figures/mwpot14/Clemens-c_free.pdf"
    )

    # -----------------------

    samples_savefilename = "output/mwpot14varyc-samples.pkl"
    if os.path.exists(samples_savefilename):
        with open(samples_savefilename, "rb") as savefile:
            s = pickle.load(savefile)
    else:
        s = su.sample(
            nsamples=100000,
            params=p_b15_cfree[0],
            fitc=True,
            c=None,
            plots=False,
            _use_emcee=True,
        )
        save_pickles(samples_savefilename, s)

    # -----------------------

    plt.figure()
    su.plot_samples(
        s, True, False, savefig="figures/mwpot14/varyc-samples-corner.pdf"
    )

    # -----------------------

    bovy_plot.bovy_print(
        axes_labelsize=17.0,
        text_fontsize=12.0,
        xtick_labelsize=15.0,
        ytick_labelsize=15.0,
    )
    su.plot_mcmc_c(
        s,
        False,
        cs,
        bf_params,
        savefig="figures/mwpot14/varyc-samples-dependence.pdf",
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
    sortedc = numpy.array(sorted(s[7]))
    plt.title(
        "2.5%% and 0.5%% lower limits: %.2f, %.2f"
        % (
            sortedc[int(numpy.floor(0.025 * len(sortedc)))],
            sortedc[int(numpy.floor(0.005 * len(sortedc)))],
        )
    )
    plt.savefig("figures/mwpot14/varyc-samples-shape_hist.pdf")

    # -------------------------------------------
    # Also fitting $R_0$ and $V_c(R_0)$

    plt.figure(figsize=(16, 5))
    p_b15_voro = su.fit(
        fitc=True,
        c=None,
        fitvoro=True,
        plots="figures/mwpot14/varyc-fitvoro-samples.pdf",
    )

    # -----------------------

    samples_savefilename = "output/mwpot14varyc-fitvoro-samples.pkl"
    if os.path.exists(samples_savefilename):
        with open(samples_savefilename, "rb") as savefile:
            s = pickle.load(savefile)
    else:
        s = su.sample(
            nsamples=100000,
            params=p_b15_voro[0],
            fitc=True,
            c=None,
            plots=False,
            fitvoro=True,
        )
        save_pickles(samples_savefilename, s)

    # -----------------------

    plt.figure()
    su.plot_samples(
        s,
        True,
        True,
        savefig="figures/mwpot14/varyc-fitvoro-samples-corner.pdf",
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
        savefig="figures/mwpot14/varyc-fitvoro-samples-dependence.pdf",
    )

    if save_figures:
        plt.savefig(
            os.path.join(
                os.getenv("PAPERSDIR"),
                "2016-mwhalo-shape",
                "mwpot14-varyc.pdf",
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
    sortedc = numpy.array(sorted(s[9]))
    plt.title(
        "2.5%% and 0.5%% lower limits: %.2f, %.2f"
        % (
            sortedc[int(numpy.floor(0.025 * len(sortedc)))],
            sortedc[int(numpy.floor(0.005 * len(sortedc)))],
        )
    )
    plt.savefig("figures/mwpot14/varyc-fitvoro-samples-shape_hist.pdf")

# /if


###############################################################################
# END
