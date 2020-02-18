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

from scipy import integrate

# matplotlib
import matplotlib.pyplot as plt

# galpy
from galpy import potential
from galpy.util import (
    bovy_plot,
    save_pickles,
    bovy_conversion,
)

# PROJECT-SPECIFIC
import script_util as su

# fmt: off
import sys; sys.path.insert(0, "../../../")
# fmt: on
from src import MWPotential2014Likelihood


###############################################################################
# PARAMETERS

numpy.random.seed(1)  # set random number seed. TODO use numpy1.8 generator

save_figures = False  # TODO needs papers-directory

_REFR0 = MWPotential2014Likelihood._REFR0
_REFV0 = MWPotential2014Likelihood._REFV0


###############################################################################
# Command Line
###############################################################################


if __name__ == "__main__":

    # -----------------------
    # Adding in the force measurements from Pal 5 *and* GD-1; also fitting
    # $R_0$ and $V_c(R_0)$

    plt.figure(figsize=(16, 5))
    p_b15_pal5gd1_voro = su.fit(
        fitc=True,
        c=None,
        addpal5=True,
        addgd1=True,
        fitvoro=True,
        mc16=True,
        plots="figures/combo_pal5_gd1/fit.pdf",
    )

    # -----------------------

    samples_savefilename = "mwpot14varyc-fitvoro-pal5gd1-samples.pkl"
    if os.path.exists(samples_savefilename):
        with open(samples_savefilename, "rb") as savefile:
            s = pickle.load(savefile)
    else:
        s = su.sample(
            nsamples=100000,
            params=p_b15_pal5gd1_voro[0],
            fitc=True,
            c=None,
            plots="figures/combo_pal5_gd1/mwpot14varyc-fitvoro-pal5gd1-samples.pdf",
            mc16=True,
            addpal5=True,
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
        addpal5=True,
        addgd1=True,
        figures="figures/combo_pal5_gd1/mwpot14varyc-fitvoro-pal5gd1-samples-corner.pdf",
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
                fitc=False,
                c=c,
                plots="figures/combo_pal5_gd1/mwpot14varyc-bf-fit.pdf",
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
        savefig="figures/combo_pal5_gd1/mwpot14varyc-bf-combo_pal5_gd1-dependence.pdf",
    )
    if save_figures:
        plt.savefig(
            os.path.join(
                os.getenv("PAPERSDIR"),
                "2016-mwhalo-shape",
                "mwpot14-varyc-wp5g1.pdf",
            ),
            bbox_inches="tight",
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
        xrange=[0.5, 1.5],
        normed=True,
    )
    plt.savefig(
        "figures/combo_pal5_gd1/mwpot14varyc-bf-combo_pal5_gd1-shape_hist.pdf"
    )

    with open("output/fit_potential_combo-pal5-gd1.txt", "wb") as file:
        sortedc = numpy.array(sorted(s[cindx][-50000:]))
        file.write(
            "2.5%% and 0.5%% lower limits: %.3f, %.3f"
            % (
                sortedc[int(numpy.floor(0.025 * len(sortedc)))],
                sortedc[int(numpy.floor(0.005 * len(sortedc)))],
            )
        )
        file.write(
            "2.5%% and 0.5%% upper limits: %.3f, %.3f"
            % (
                sortedc[int(numpy.floor(0.975 * len(sortedc)))],
                sortedc[int(numpy.floor(0.995 * len(sortedc)))],
            )
        )
        file.write(
            "Median, 68%% confidence: %.3f, %.3f, %.3f"
            % (
                numpy.median(sortedc),
                sortedc[int(numpy.floor(0.16 * len(sortedc)))],
                sortedc[int(numpy.floor(0.84 * len(sortedc)))],
            )
        )
        file.write(
            "Mean, std. dev.: %.2f,%.2f"
            % (numpy.mean(sortedc), numpy.std(sortedc),)
        )

    # -----------------------
    # What is the constraint on the mass of the halo?

    tR = 20.0 / _REFR0
    skip = 1
    hmass = []
    for sa in tqdm(s.T[::skip]):
        pot = MWPotential2014Likelihood.setup_potential(
            sa,
            sa[-1],
            True,
            False,
            _REFR0 * sa[8],
            _REFV0 * sa[7],
            fitvoro=True,
        )
        hmass.append(
            -integrate.quad(
                lambda x: tR ** 2.0
                * potential.evaluaterforces(
                    pot[2], tR * x, tR * numpy.sqrt(1.0 - x ** 2.0), phi=0.0
                ),
                0.0,
                1.0,
            )[0]
            * bovy_conversion.mass_in_1010msol(_REFV0, _REFR0)
            / 10.0
        )
    hmass = numpy.array(hmass)

    with open(
        "output/fit_potential_combo-pal5-gd1.txt", "ab"
    ) as file:  # append

        file.write("\nMass Constraints:")

        sortedhm = numpy.array(sorted(hmass))
        file.write(
            "2.5%% and 0.5%% lower limits: %.2f, %.2f"
            % (
                sortedhm[int(numpy.floor(0.025 * len(sortedhm)))],
                sortedhm[int(numpy.floor(0.005 * len(sortedhm)))],
            )
        )
        file.write(
            "2.5%% and 0.5%% upper limits: %.2f, %.2f"
            % (
                sortedhm[int(numpy.floor(0.975 * len(sortedhm)))],
                sortedhm[int(numpy.floor(0.995 * len(sortedhm)))],
            )
        )
        file.write(
            "Median, 68%% confidence: %.2f, %.2f, %.2f"
            % (
                numpy.median(sortedhm),
                sortedhm[int(numpy.floor(0.16 * len(sortedhm)))],
                sortedhm[int(numpy.floor(0.84 * len(sortedhm)))],
            )
        )

    # -----------------------

    bovy_plot.scatterplot(
        hmass,
        s[-1, ::skip],
        "k,",
        onedhists=True,
        bins=31,
        xrange=[0.5, 1.5],
        yrange=[0.5, 1.5],
        xlabel=r"$M_{\mathrm{halo}} (r<20\,\mathrm{kpc})\,(M_\odot)$",
        ylabel=r"$c/a$",
    )
    plt.savefig("figures/combo_pal5_gd1/scatterplot.pdf")

    # -----------------------


# /if


###############################################################################
# END
