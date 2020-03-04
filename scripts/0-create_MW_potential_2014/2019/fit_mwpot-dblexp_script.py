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
import matplotlib
import matplotlib.pyplot as plt

# galpy
# from galpy import potential
from galpy.util import (
    bovy_plot,
    save_pickles,
)  # bovy_conversion

# CUSTOM

# PROJECT-SPECIFIC
import script_util as su


###############################################################################
# PARAMETERS

numpy.random.seed(1)  # set random number seed. TODO use numpy1.8 generator

matplotlib.use("Agg")
save_figures = False  # TODO needs papers-directory


###############################################################################
# Command Line
###############################################################################


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Using an exponential disk instead of a Miyamoto-Nagai disk

    # -----------------------
    # $c=1$:

    plt.figure(figsize=(16, 5))
    p_exp = su.fit(
        fitc=False,
        c=1.0,
        dblexp=True,
        plots="figures/mwpot_dblexp/Clemens-c_1.pdf",
    )

    # -----------------------
    # $c=0.5$:

    plt.figure(figsize=(16, 5))
    p_exp = su.fit(
        fitc=False,
        c=0.5,
        dblexp=True,
        plots="figures/mwpot_dblexp/Clemens-c_0p5.pdf",
    )

    # -----------------------
    # $c=1.5$:

    plt.figure(figsize=(16, 5))
    p_exp = su.fit(
        fitc=False,
        c=1.5,
        dblexp=True,
        plots="figures/mwpot_dblexp/Clemens-c_1p5.pdf",
    )

    # -----------------------
    # leave c free

    plt.figure(figsize=(16, 5))
    p_exp_cfree = su.fit(
        fitc=True,
        c=None,
        dblexp=True,
        plots="figures/mwpot_dblexp/Clemens-c_free.pdf",
    )

    # -----------------------

    bf_savefilename = "output/mwpot14varyc-dblexp-bf.pkl"
    if os.path.exists(bf_savefilename):
        with open(bf_savefilename, "rb") as savefile:
            cs = pickle.load(savefile)
            bf_params = pickle.load(savefile)
    else:
        cs = numpy.arange(0.5, 4.1, 0.1)
        bf_params = []
        for c in tqdm(cs):
            dum = su.fit(
                fitc=False,
                c=c,
                dblexp=True,
                plots="figures/mwpot_dblexp/mwpot14varyc-dblexp-bf-fit.pdf",
            )
            bf_params.append(dum[0])
        save_pickles(bf_savefilename, cs, bf_params)

    # -----------------------

    samples_savefilename = "output/mwpot14varyc-dblexp-samples.pkl"
    if os.path.exists(samples_savefilename):
        with open(samples_savefilename, "rb") as savefile:
            s = pickle.load(savefile)
    else:
        s = su.sample(
            nsamples=100000,
            params=p_exp_cfree[0],
            fitc=True,
            c=None,
            plots="figures/mwpot_dblexp/mwpot14varyc-dblexp-samples.pdf",
            dblexp=True,
        )
        save_pickles(samples_savefilename, s)

    # -----------------------

    plt.figure()
    su.plot_samples(
        s,
        True,
        False,
        savefig="figures/mwpot_dblexp/varyc-dblexp-samples-corner.pdf",
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
        False,
        cs,
        bf_params,
        savefig="figures/mwpot_dblexp/varyc-dblexp-samples-dependence.pdf",
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
    plt.savefig("figures/mwpot_dblexp/varyc-dblexp-samples-shape_hist.pdf")

    # -----------------------
    # Also fitting $R_0$ and $V_c(R_0)$

    plt.figure(figsize=(16, 5))
    p_exp_cfree_voro = su.fit(
        fitc=True,
        c=None,
        dblexp=True,
        fitvoro=True,
        plots="figures/mwpot_dblexp/fitvoro-samples.pdf",
    )

    # -----------------------

    samples_savefilename = "output/mwpot14varyc-dblexp-fitvoro-samples.pkl"
    if os.path.exists(samples_savefilename):
        with open(samples_savefilename, "rb") as savefile:
            s = pickle.load(savefile)
    else:
        s = su.sample(
            nsamples=100000,
            params=p_exp_cfree_voro[0],
            fitc=True,
            c=None,
            plots="figures/mwpot_dblexp/mwpot14varyc-dblexp-fitvoro-samples.pdf",
            dblexp=True,
            fitvoro=True,
        )
        save_pickles(samples_savefilename, s)

    # -----------------------

    plt.figure()
    su.plot_samples(
        s,
        True,
        True,
        savefig="figures/mwpot_dblexp/varyc-dblexp-fitvoro-samples-corner.pdf",
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
        savefig="figures/mwpot_dblexp/varyc-dblexp-fitvoro-samples-dependence.pdf",
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
    plt.savefig(
        "figures/mwpot_dblexp/varyc-dblexp-fitvoro-samples-samples-shape_hist.pdf"
    )

    # -----------------------
    # Also adding in a gas disk (and still also fitting $R_0$ and $V_c(R_0)$)

    plt.figure(figsize=(16, 5))
    p_exp_cfree_voro_wgas = su.fit(
        fitc=True,
        c=None,
        dblexp=True,
        fitvoro=True,
        addgas=True,
        plots="figures/mwpot_dblexp/varyc-dblexp-fitvoro-addgas.pdf",
    )

    # -----------------------

    samples_savefilename = "mwpot14varyc-dblexp-fitvoro-addgas-samples.pkl"
    if os.path.exists(samples_savefilename):
        with open(samples_savefilename, "rb") as savefile:
            s = pickle.load(savefile)
    else:
        s = su.sample_multi(
            nsamples=100000,
            params=p_exp_cfree_voro_wgas[0],
            fitc=True,
            c=None,
            plots="figures/mwpot_dblexp/mwpot14varyc-dblexp-fitvoro-addgas-samples.pdf",
            dblexp=True,
            fitvoro=True,
            addgas=True,
        )
        save_pickles(samples_savefilename, s)

    # -----------------------

    plt.figure()
    su.plot_samples(
        s,
        True,
        True,
        savefig="figures/mwpot_dblexp/varyc-dblexp-fitvoro-addgas-samples-corner.pdf",
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
        savefig="figures/mwpot_dblexp/varyc-dblexp-fitvoro-addgas-samples-dependence.pdf",
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
    plt.savefig(
        "figures/mwpot_dblexp/varyc-dblexp-fitvoro-addgas-samples-shape_hist.pdf"
    )

# /if


###############################################################################
# END
