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

    # -----------------------
    # Adding in the force measurements from GD-1; also fitting $R_0$ and $V_c(R_0)$

    plt.figure(figsize=(16, 5))
    p_b15_gd1_voro = su.fit(
        fitc=True,
        c=None,
        addgd1=True,
        fitvoro=True,
        mc16=True,
        plots="figures/gd1/fit.pdf",
    )

    # -----------------------

    samples_savefilename = "mwpot14varyc-fitvoro-gd1-samples.pkl"
    if os.path.exists(samples_savefilename):
        with open(samples_savefilename, "rb") as savefile:
            s = pickle.load(savefile)
    else:
        s = su.sample(
            nsamples=100000,
            params=p_b15_gd1_voro[0],
            fitc=True,
            c=None,
            plots="figures/gd1/mwpot14varyc-fitvoro-gd1-samples.pdf",
            addgd1=True,
            fitvoro=True,
        )
        save_pickles(samples_savefilename, s)

    # -----------------------

    plt.figure()
    su.plot_samples(
        s,
        True,
        True,
        addgd1=True,
        savefig="figures/mwpot14varyc-fitvoro-gd1-samples-corner.pdf",
    )

    # -----------------------

    bf_savefilename = "output/mwpot14varyc-bf.pkl"  # should already exist
    if os.path.exists(bf_savefilename):
        with open(bf_savefilename, "rb") as savefile:
            cs = pickle.load(savefile)
            bf_params = pickle.load(savefile)
    else:
        cs = numpy.arange(0.5, 4.1, 0.1)
        bf_params = []
        for c in tqdm(cs):
            dum = su.fit(
                fitc=False, c=c, plots="figures/gd1/mwpot14varyc-bf-fit.pdf"
            )
            bf_params.append(dum[0])
        save_pickles(bf_savefilename, cs, bf_params)

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
        savefig="figures/mwpot14varyc-fitvoro-gd1-samples-dependence.pdf",
    )

    # -----------------------

    plt.figure(figsize=(4, 4))
    cindx = 9
    dum = bovy_plot.bovy_hist(
        s[cindx],
        bins=36,
        histtype="step",
        lw=2.0,
        xlabel=r"$c/a$",
        xrange=[0.5, 2.5],
        normed=True,
    )
    plt.savefig("figures/gd1/mwpot14varyc-fitvoro-gd1-shape_hist")

    # -----------------------

    with open("output/fit_potential_gd1.txt", "wb") as file:

        sortedc = numpy.array(sorted(s[cindx]))
        file.write(
            "2.5%% and 0.5%% lower limits: %.2f, %.2f"
            % (
                sortedc[int(numpy.floor(0.025 * len(sortedc)))],
                sortedc[int(numpy.floor(0.005 * len(sortedc)))],
            )
        )
        file.write(
            "2.5%% and 0.5%% upper limits: %.2f, %.2f"
            % (
                sortedc[int(numpy.floor(0.975 * len(sortedc)))],
                sortedc[int(numpy.floor(0.995 * len(sortedc)))],
            )
        )
        file.write(
            "Median, 68%% confidence: %.2f, %.2f, %.2f"
            % (
                numpy.median(sortedc),
                sortedc[int(numpy.floor(0.16 * len(sortedc)))],
                sortedc[int(numpy.floor(0.84 * len(sortedc)))],
            )
        )

    # -----------------------


# /if


###############################################################################
# END
