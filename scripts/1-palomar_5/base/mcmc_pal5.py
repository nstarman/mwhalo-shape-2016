# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   : mcmc_pal5
# PROJECT : Pal 5 update MW pot constraints
#
# ----------------------------------------------------------------------------

# Docstring
"""MCMC analysis routines of Palomar 5.

Routing Listings
----------------
load_samples
find_starting_point
lnp
get_options

References
----------
https://github.com/jobovy/mwhalo-shape-2016

"""

__author__ = "Jo Bovy"
__copyright__ = "Copyright 2016, 2020, "
__maintainer__ = "Nathaniel Starkman"

__all__ = [
    "_DATADIR",
    "load_samples",
    "find_starting_point",
    "lnp",
    "get_options",
]


###############################################################################
# IMPORTS

# GENERAL
import os
import os.path
import copy
import time
import pickle
import csv
from argparse import ArgumentParser, Namespace
from typing import Optional

import numpy as np
from scipy.special import logsumexp
import emcee

# PROJECT-SPECIFIC
# fmt: off
import sys; sys.path.append('../../../')
# fmt: on
from src import pal5_util


###############################################################################
# PARAMETERS

_DATADIR = os.getenv("DATADIR")


###############################################################################
# CODE
###############################################################################


def load_samples(options: Namespace):
    """Load Samples.

    Parameters
    ----------
    options: Namespace

    Returns
    -------
    samples: array

    """
    print(options.samples_savefilename)
    if os.path.exists(options.samples_savefilename):
        with open(options.samples_savefilename, "rb") as savefile:
            samples = pickle.load(savefile)
    else:
        raise IOError(
            f"File {options.samples_savefilename} that is supposed "
            "to hold the potential samples does not exist"
        )

    return samples


# /def


# ------------------------------------------------------------------------


def find_starting_point(
    options: Namespace,
    pot_params: list,
    dist: float,
    pmra: float,
    pmdec: float,
    sigv: float,
):
    """Find Starting Point.

    Find a decent starting point
    useTM to speed this up, bc it doesn't matter much

    Parameters
    ----------
    options: Namespace
    pot_params: list
    dist: float
    pmra: float
    pmdec: float
    sigv: float

    Returns
    -------
    maxlk: array

    """
    # start with a prediction
    interpcs = [0.65, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 1.65]
    cs = np.arange(0.7, 1.61, 0.01)
    pal5varyc_like = pal5_util.predict_pal5obs(
        pot_params,
        cs,
        dist=dist,
        pmra=pmra,
        pmdec=pmdec,
        sigv=sigv,
        td=options.td,
        ro=options.ro,
        vo=220.0,
        interpk=1,
        interpcs=interpcs,
        useTM=True,
        trailing_only=True,
        verbose=False,
    )

    # load data against which to compare
    pos_radec, rvel_ra = pal5_util.pal5_data_2016()

    if options.fitsigma:
        lnlike = np.sum(
            pal5_util.pal5_lnlike(
                pos_radec,
                rvel_ra,
                pal5varyc_like[0],
                pal5varyc_like[1],
                pal5varyc_like[2],
                pal5varyc_like[3],
                pal5varyc_like[4],
                pal5varyc_like[5],
                pal5varyc_like[6],
            )[:, :3:2],
            axis=1,
        )
    else:
        # For each one, move the track up and down a little
        # to simulate sig changes
        deco = np.linspace(-0.5, 0.5, 101)
        lnlikes = np.zeros((len(cs), len(deco))) - 100000000000000000.0
        for jj, do in enumerate(deco):
            tra = pal5varyc_like[0]
            tra[:, :, 1] += do
            lnlikes[:, jj] = pal5_util.pal5_lnlike(
                pos_radec,
                rvel_ra,
                tra,
                pal5varyc_like[1],
                pal5varyc_like[2],
                pal5varyc_like[3],
                pal5varyc_like[4],
                pal5varyc_like[5],
                pal5varyc_like[6],
            )[:, 0]
        lnlike = np.amax(lnlikes, axis=1)

    return cs[np.argmax(lnlike)]


# /def


# ------------------------------------------------------------------------


def lnp(p: list, pot_params: list, options: Namespace):
    """Log Likelihood.

    Parameters
    ----------
    p: list
    pot_params: list
    options: Namespace

    Returns
    -------
    ndarray

    """
    # warnings.filterwarnings(
    #     "ignore", message="Using C implementation to integrate orbits"
    # )
    # p=[c,vo/220,dist/22.,pmo_parallel,pmo_perp] and ln(sigv) if fitsigma

    # Parameters
    c = p[0]
    vo = p[1] * pal5_util._REFV0
    dist = p[2] * 22.0  # TODO in units of 22 kpc, change to 20
    pmra = -2.296 + p[3] + p[4]
    pmdecpar = 2.257 / 2.296
    pmdecperp = -2.296 / 2.257
    pmdec = -2.257 + p[3] * pmdecpar + p[4] * pmdecperp

    if options.fitsigma:
        sigv = 0.4 * np.exp(p[5])
    else:
        sigv = 0.4

    # Priors
    if c < 0.5:
        return -100000000000000000.0
    elif c > 2.0:
        return -10000000000000000.0
    elif vo < 200:
        return -10000000000000000.0
    elif vo > 250:
        return -10000000000000000.0
    elif dist < 19.0:
        return -10000000000000000.0
    elif dist > 24.0:
        return -10000000000000000.0
    elif options.fitsigma and sigv < 0.1:
        return -10000000000000000.0
    elif options.fitsigma and sigv > 1.0:
        return -10000000000000000.0

    # Setup the model
    trailing_only = False  # NOTE change
    pal5varyc_like = pal5_util.predict_pal5obs(
        pot_params,
        c,
        singlec=True,
        dist=dist,
        pmra=pmra,
        pmdec=pmdec,
        ro=options.ro,
        vo=vo,
        trailing_only=trailing_only,
        verbose=False,
        sigv=sigv,
        td=options.td,
        useTM=False,
        nTrackChunks=8,
    )

    pos_radec, rvel_ra = pal5_util.pal5_total_data()  # NOTE changed

    if options.fitsigma:
        lnlike = pal5_util.pal5_lnlike(
            pos_radec,
            rvel_ra,
            pal5varyc_like[0],
            pal5varyc_like[1],
            pal5varyc_like[2],
            pal5varyc_like[3],
            pal5varyc_like[4],
            pal5varyc_like[5],
            pal5varyc_like[6],
            trailing_only=trailing_only,
        )
        if not pal5varyc_like[7]:
            addllnlike = -15.0  # penalize
        else:
            addllnlike = 0.0
        # print addllnlike, pal5varyc_like[7]
        # sys.stdout.flush()
        # NOTE: Fritz and Kallivaylil paper likelihood
        return (
            lnlike[0, 0]
            + lnlike[0, 2]
            + addllnlike
            + -0.5 * (pmra + 2.296) ** 2.0 / 0.186 ** 2.0
            - 0.5 * (pmdec + 2.257) ** 2.0 / 0.181 ** 2.0
        )

    # If not fitsigma, move the track up and down a little
    # to simulate sig changes
    deco = np.linspace(-0.5, 0.5, 101)
    lnlikes = np.zeros(len(deco)) - 100000000000000000.0
    for jj, do in enumerate(deco):
        tra = copy.deepcopy(pal5varyc_like[0])
        tra[:, :, 1] += do
        lnlikes[jj] = pal5_util.pal5_lnlike(
            pos_radec,
            rvel_ra,
            tra,
            pal5varyc_like[1],
            pal5varyc_like[2],
            pal5varyc_like[3],
            pal5varyc_like[4],
            pal5varyc_like[5],
            pal5varyc_like[6],
        )[0, 0]

    # Calculate and return a log-likelihood
    return (
        logsumexp(lnlikes)
        + pal5_util.pal5_lnlike(
            pos_radec,
            rvel_ra,
            pal5varyc_like[0],
            pal5varyc_like[1],
            pal5varyc_like[2],
            pal5varyc_like[3],
            pal5varyc_like[4],
            pal5varyc_like[5],
            pal5varyc_like[6],
        )[0, 2]
        - 0.5 * (pmra + 2.296) ** 2.0 / 0.186 ** 2.0
        - 0.5 * (pmdec + 2.257) ** 2.0 / 0.181 ** 2.0
    )


# /def


###############################################################################
# Command Line
###############################################################################


def make_parser():
    """Command-line Options."""
    usage = "usage: %prog [options]"
    parser = ArgumentParser(
        usage=usage, description="MCMC analysis routines of Palomar 5"
    )
    # Potential parameters
    parser.add_argument(
        "--bf_b15",
        action="store_true",
        dest="bf_b15",
        default=False,
        help="If set, use the best-fit to the MWPotential2014 data",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        default=1,
        type="int",
        help="seed for everything except for potential",
    )
    parser.add_argument(
        "--fitsigma",
        action="store_true",
        dest="fitsigma",
        default=False,
        help="If set, fit for the velocity-dispersion parameter",
    )
    parser.add_argument(
        "--dt",
        dest="dt",
        default=10.0,
        type="float",
        help="Run MCMC for this many minutes",
    )
    parser.add_argument(
        "-i",
        dest="pindx",
        default=None,
        type="int",
        help="Index into the potential samples to consider",
    )
    parser.add_argument(
        "--ro",
        dest="ro",
        default=pal5_util._REFR0,
        type="float",
        help="Distance to the Galactic center in kpc",
    )
    parser.add_argument(
        "--td",
        dest="td",
        default=5.0,
        type="float",
        help="Age of the stream in Gyr",
    )
    parser.add_argument(
        "--samples_savefilename",
        dest="samples_savefilename",
        default=(
            "../../0-create_MW_potential_2014/"
            "latest/output/mwpot14varyc-samples.pkl"
        ),
        help="Name of the file that contains the potential samples",
    )
    # Output file
    parser.add_argument(
        "-o",
        dest="outfilename",
        default=None,
        help="Name of the file that will hold the output",
    )
    # Multi-processing
    parser.add_argument(
        "-m",
        dest="multi",
        default=1,
        type="int",
        help="Number of CPUs to use for streamdf setup",
    )
    return parser


# /def


# ------------------------------------------------------------------------


def main(args: Optional[list] = None, opts: Optional[Namespace] = None):
    """MCMC Pal 5.

    Parameters
    ----------
    args : list, optional
        an optional single argument that holds the sys.argv list,
        except for the script name (e.g., argv[1:])

    """
    if opts is not None and args is None:
        options = opts
    else:
        parser = make_parser()
        options = parser.parse_args(args)

    # Set random seed for potential selection
    np.random.seed(1)

    # Load potential parameters
    if options.bf_b15:
        pot_params = [
            0.60122692,
            0.36273147,
            -0.97591502,
            -3.34169377,
            0.71877924,
            -0.01519337,
            -0.01928001,
        ]
    else:
        pot_samples = load_samples(options)
        rndindx = np.random.permutation(pot_samples.shape[1])[options.pindx]
        pot_params = pot_samples[:, rndindx]
    print(pot_params)

    # Now set the seed for the MCMC
    np.random.seed(options.seed)
    nwalkers = 10 + 2 * options.fitsigma

    # For a fiducial set of parameters, find a good fit to use as the starting
    # point
    all_start_params = np.zeros((nwalkers, 5 + options.fitsigma))
    start_lnprob0 = np.zeros(nwalkers)
    if not os.path.exists(options.outfilename):
        pmra = -2.296
        pmdec = -2.257
        dist = 23.2
        # cstart= find_starting_point(options,pot_params,dist,pmra,pmdec,0.4)
        cstart = 1.0
        if cstart > 1.15:
            cstart = 1.15  # Higher c doesn't typically really work
        if options.fitsigma:
            start_params = np.array([cstart, 1.0, dist / 22.0, 0.0, 0.0, 0.0])
            step = np.array([0.05, 0.05, 0.05, 0.05, 0.01, 0.05])
        else:
            start_params = np.array([cstart, 1.0, dist / 22.0, 0.0, 0.0])
            step = np.array([0.05, 0.05, 0.05, 0.05, 0.01])
        nn = 0
        print("walker: ", end="")
        while nn < nwalkers:
            print(nn, end=", ")
            all_start_params[nn] = (
                start_params + np.random.normal(size=len(start_params)) * step
            )
            start_lnprob0[nn] = lnp(all_start_params[nn], pot_params, options)
            if start_lnprob0[nn] > -1000000.0:
                print(all_start_params[nn], start_lnprob0[nn])
            if start_lnprob0[nn] > -1000000.0:
                nn += 1
    else:
        print("continuing chain")
        # Get the starting point from the output file
        with open(options.outfilename, "r") as savefile:
            all_lines = savefile.readlines()
        for nn in range(nwalkers):
            lastline = all_lines[-1 - nn]
            tstart_params = np.array([float(s) for s in lastline.split(",")])
            start_lnprob0[nn] = tstart_params[-1]
            all_start_params[nn] = tstart_params[:-1]

    # Output
    # pdb.set_trace()
    if os.path.exists(options.outfilename):
        outfile = open(options.outfilename, "a")
    else:
        # Setup the file
        outfile = open(options.outfilename, "w")
        outfile.write(
            "# potparams:%.8f,%.8f,%.8f,%.8f,%.8f\n"
            % (
                pot_params[0],
                pot_params[1],
                pot_params[2],
                pot_params[3],
                pot_params[4],
            )
        )
        for nn in range(nwalkers):
            if options.fitsigma:
                outfile.write(
                    "%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n"
                    % (
                        all_start_params[nn, 0],
                        all_start_params[nn, 1],
                        all_start_params[nn, 2],
                        all_start_params[nn, 3],
                        all_start_params[nn, 4],
                        all_start_params[nn, 5],
                        start_lnprob0[nn],
                    )
                )
            else:
                outfile.write(
                    "%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n"
                    % (
                        all_start_params[nn, 0],
                        all_start_params[nn, 1],
                        all_start_params[nn, 2],
                        all_start_params[nn, 3],
                        all_start_params[nn, 4],
                        start_lnprob0[nn],
                    )
                )
        outfile.flush()
    # outwriter = csv.writer(outfile, delimiter=",")

    # Run MCMC
    sampler = emcee.EnsembleSampler(
        nwalkers,
        all_start_params.shape[1],
        lnp,
        args=(pot_params, options),
        threads=options.multi,
    )

    rstate0 = np.random.mtrand.RandomState().get_state()
    start = time.time()
    while time.time() < start + options.dt * 60.0:
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
        for nn in range(nwalkers):
            if options.fitsigma:
                outfile.write(
                    "%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n"
                    % (
                        all_start_params[nn, 0],
                        all_start_params[nn, 1],
                        all_start_params[nn, 2],
                        all_start_params[nn, 3],
                        all_start_params[nn, 4],
                        all_start_params[nn, 5],
                        start_lnprob0[nn],
                    )
                )
            else:
                outfile.write(
                    "%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n"
                    % (
                        all_start_params[nn, 0],
                        all_start_params[nn, 1],
                        all_start_params[nn, 2],
                        all_start_params[nn, 3],
                        all_start_params[nn, 4],
                        start_lnprob0[nn],
                    )
                )
        outfile.flush()
    outfile.close()


# /def


# ------------------------------------------------------------------------

if __name__ == "__main__":

    import sys

    main(sys.argv[1:])

# /if


###############################################################################
# END
