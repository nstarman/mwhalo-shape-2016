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

__all__ = ["main"]


###############################################################################
# IMPORTS

# GENERAL
import os
import os.path
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from typing import Union, Optional

# matplotlib
import matplotlib.pyplot as plt

# galpy
from galpy.util import (
    bovy_plot,
    save_pickles,
)

# PROJECT-SPECIFIC

# fmt: off
import sys; sys.path.insert(0, "../../../")
# fmt: on
from pal5_constrain_mwhalo_shape import mw_pot
import pal5_constrain_mwhalo_shape.scripts.create_MW_potential_2014.script_util as su


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
        prog="fit_potential_pal5_script",
        description="Fit Pal5 to MW potential.",
        add_help=~inheritable,
        conflict_handler="resolve" if ~inheritable else "error",
    )
    parser.add_argument(
        "-f",
        "--figure",
        default="figures/pal5/",
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
    """Fit Pal5 to MW Potential Script Function.

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

    # -------------------------------------------------------------------------
    # Adding in the force measurements from Pal 5
    # also fitting $R_0$ and $V_c(R_0)$

    plt.figure(figsize=(16, 5))
    p_b15_pal5_voro = mw_pot.fit(
        fitc=True,
        c=None,
        addpal5=True,
        fitvoro=True,
        mc16=True,
        plots=fpath + "fit.pdf",
    )

    # -----------------------

    samples_savefilename = opath + "mwpot14varyc-fitvoro-pal5-samples.pkl"
    if os.path.exists(samples_savefilename):
        with open(samples_savefilename, "rb") as savefile:
            s = pickle.load(savefile)
    else:
        s = mw_pot.sample(
            nsamples=100000,
            params=p_b15_pal5_voro[0],
            fitc=True,
            c=None,
            addpal5=True,
            fitvoro=True,
            plots=fpath + "mwpot14varyc-fitvoro-pal5-fit.pdf",
        )
        save_pickles(samples_savefilename, s)

    # -----------------------

    plt.figure()
    mw_pot.plot_samples(
        s,
        True,
        True,
        addpal5=True,
        savefig=fpath + "mwpot14varyc-fitvoro-pal5-samples-corner.pdf",
    )

    # -----------------------

    bf_savefilename = opath + "mwpot14varyc-bf.pkl"  # should already exist
    if os.path.exists(bf_savefilename):
        with open(bf_savefilename, "rb") as savefile:
            cs = pickle.load(savefile)
            bf_params = pickle.load(savefile)
    else:
        cs = np.arange(0.5, 4.1, 0.1)
        bf_params = []
        for c in tqdm(cs):
            dum = mw_pot.fit(fitc=False, c=c, plots=fpath + "mwpot14varyc-bf-fit.pdf")
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
        s, True, cs, bf_params, savefig=fpath + "mwpot14varyc-bf-pal5-dependence.pdf",
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
    plt.savefig(fpath + "mwpot14varyc-bf-pal5-shape_hist.pdf")

    # -----------------------

    sortedc = np.array(sorted(s[cindx]))

    with open(opath + "fit_potential_pal5.txt", "w") as file:

        file.write(
            "2.5%% and 0.5%% lower limits: %.2f, %.2f"
            % (
                sortedc[int(np.floor(0.025 * len(sortedc)))],
                sortedc[int(np.floor(0.005 * len(sortedc)))],
            )
        )
        file.write(
            "2.5%% and 0.5%% upper limits: %.2f, %.2f"
            % (
                sortedc[int(np.floor(0.975 * len(sortedc)))],
                sortedc[int(np.floor(0.995 * len(sortedc)))],
            )
        )
        file.write(
            "Median, 68%% confidence: %.2f, %.2f, %.2f"
            % (
                np.median(sortedc),
                sortedc[int(np.floor(0.16 * len(sortedc)))],
                sortedc[int(np.floor(0.84 * len(sortedc)))],
            )
        )

    # -----------------------


# /if

# ------------------------------------------------------------------------

if __name__ == "__main__":

    main(args=None, opts=None)

# /if


###############################################################################
# END
