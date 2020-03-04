# -*- coding: utf-8 -*-

"""Utility Functions for the MCMC."""

__author__ = ["Jo Bovy", "Nathaniel Starkman"]


###############################################################################
# IMPORTS

# GENERAL
import os
import pickle
import numpy as np
from numpy.linalg import norm

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

# galpy
from galpy.orbit import Orbit
from galpy.util.bovy_coords import lb_to_radec
from galpy.util import bovy_plot

# CUSTOM
# from astroPHD.util import ObjDict, LogFile
# from astroPHD.util.multi import parallel_map
# from astroPHD.select import inRange


# PROJECT-SPECIFIC
from . import MWPotential2014Likelihood
from . import pal5_util


###############################################################################
# PARAMETERS

_R0 = MWPotential2014Likelihood._REFR0
_V0 = MWPotential2014Likelihood._REFV0

# Some parameters defining the proper motion direction in Bovy et al. (2016)
_PMDECPAR = 2.257 / 2.296
_PMDECPERP = -2.296 / 2.257

# _LOGFILE = LogFile(verbose=2, header=False)


###############################################################################
# CODE
###############################################################################


def determine_nburn(
    filename="pal5_mcmc/mwpot14-fitsigma-0.dat",
    threshold=0.1,
    skip=50,
    nwalkers=12,
    return_nsamples=False,
):
    """Function to determine an appropriate nburn for a given chain.

    Leave defaults as is for consistency with Bovy et al. (2016)

    """
    # Load the data
    data = np.loadtxt(filename, comments="#", delimiter=",")
    lndata = np.reshape(data[:, -1], (len(data[:, 5]) // nwalkers, nwalkers))
    # Perform a running diff wrt skip less
    diff = lndata - np.roll(lndata, skip, axis=0)
    diff[:skip] = -100.0  # Make sure it's not within the first hundred
    maxln = np.nanmax(lndata)

    try:
        indx = (np.fabs(np.median(diff, axis=1)) < threshold) * (
            (maxln - np.nanmax(lndata, axis=1)) < 1.25
        )

    except IndexError:
        if return_nsamples:
            return 100.0
        else:
            return np.prod(lndata.shape) - 100

    else:
        if maxln > -22.5:
            indx *= np.std(lndata, axis=1) < 3.0

        if return_nsamples:
            return len(data) - np.arange(len(lndata))[indx][0] * nwalkers

        if indx.sum() == 0:
            return 0

        else:
            return np.arange(len(lndata))[indx][0] * nwalkers


# /def


# ----------------------------------------------------------------------------


def read_mcmc_chain(
    chain, drct="pal5_mcmc", return_potparams=True, permute=False
):
    """Read one of the MCMC chains.

    Parameters
    ----------
    chain : int
        chain index (0 to 31)
    drct : string
        directory that contains the MCMC sample files
    return_potparams : bool, optional
        (default=True)
        if True, return the potential parameters
    permute : bool, optional
        (default=False)
        if True, permute the samples (useful if you want a random one)

    Returns
    -------
    potparams: list
        list of potential parameters
    chain_params: ndarray
        array of chain parameters, shape=(nsamples,nparams)=(nsamples,7)
        These parameters are: halo flattening c, Vc/_REFV0, distance/22 kpc,
        2 parameters related to Pal 5's proper motion, log(sigma_v/0.4 km/s),
        the final entry is the log likelihood of the sample

    """
    fn = os.path.join(drct, "mwpot14-fitsigma-%i.dat" % chain)
    with open(fn, "rb") as savefile:
        line1 = savefile.readline().decode("utf-8")
    potparams = [float(s) for s in (line1.split(":")[1].split(","))]

    # Now read the MCMC chain
    tnburn = determine_nburn(fn)
    tdata = np.loadtxt(fn, comments="#", delimiter=",")
    tdata = tdata[tnburn::]

    if permute:
        tdata = tdata[np.random.permutation(len(tdata))]

    if return_potparams:
        return potparams, tdata
    else:
        return tdata


# /def


# ----------------------------------------------------------------------------


def setup_model(
    potparams,
    chain_params,
    td=10.0,
    multi=None,
    nTrackChunks=8,
    verbose=False,
    ro=_R0,
    vo=_V0,
    refR0=_R0,
    refV0=_V0,
    pmdecpar=_PMDECPAR,
    pmdecperp=_PMDECPERP,
    num_t=1001,
):
    """Setup the potential + Pal 5 model.

    Parameters
    ----------
    potparams: list
        list of potential parameters
    chain_params: list or numpy.ndarray with shape (7,)
        list of Pal 5 parameters and MCMC sample specific potential parameters
    td: float, optional
        age of the stream in Gyr
    multi: int, optional
        if set, use this many CPUs in setting up the stream models
        (<= nTrackChunks)
    nTrackChunks: int, optional
        number of points along the stream to directly compute the
        stream location at (rest is interpolated)
    verbose: bool, optional
        if True, increase verbosity
    ro :
    vo :
    refR0 :
    refV0 :
    pmdecpar :
    pmdecperp :
    num_t : int (default 1001)
        number of points along each track

    Returns
    -------
    pot: galpy Potential
        the potential
    sdft: galpy.df.streamdf instance
        the trailing arm
    sdfl: galpy.df.streamdf instance
        the leading arm

    """
    # Setup potential
    tvo = chain_params[1] * refV0  # TODO should this be from pal5_util?
    pot = MWPotential2014Likelihood.setup_potential(
        potparams, chain_params[0], False, False, refR0, tvo
    )

    # Now compute the stream model for this setup
    dist = chain_params[2] * 22.0
    pmra = -2.296 + chain_params[3] + chain_params[4]
    pmdec = -2.257 + chain_params[3] * pmdecpar + chain_params[4] * pmdecperp
    vlos = -58.7
    sigv = 0.4 * np.exp(chain_params[5])
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
        td,
        ro,
        tvo,
        multi=multi,
        nTrackChunks=nTrackChunks,
        trailing_only=False,
        verbose=verbose,
        useTM=False,
        num_t=num_t,
    )

    return pot, tsdf_trailing, tsdf_leading


# /def


# ----------------------------------------------------------------------------


def pm_pal5(chain_params):
    """Compute the proper motion of Pal 5 for an MCMC sample.

    Parameters
    ----------
    chain_params: list or numpy.ndarray with shape (7,)
        list of Pal 5 parameters and MCMC sample specific potential parameters

    Returns
    -------
    pmra: float
        proper motion in RA (x cos(dec)) in mas/yr
    pmdec: float
        proper motion in Dec in mas/yr

    """
    pmra = -2.296 + chain_params[3] + chain_params[4]
    pmdec = (
        -2.257 + (_PMDECPAR * chain_params[3]) + (_PMDECPERP * chain_params[4])
    )
    return (pmra, pmdec)


# /def


###############################################################################


def get_track_RADec(*sdfs):
    """get_track_RaDec.

    Parameters
    ----------
    *sdfs : streamdf(s)

    Returns
    -------
    trackRADec(s) : (n, 2) array, or list of trackRADecs
        the track in ra, dec of the streamdf

    """
    if len(sdfs) == 0:
        raise ValueError

    trackRADec = []

    for sdf in sdfs:
        trackRADec.append(
            lb_to_radec(
                sdf._interpolatedObsTrackLB[:, 0],
                sdf._interpolatedObsTrackLB[:, 1],
                degree=True,
            )
        )
    if len(trackRADec) == 1:
        return trackRADec[0]
    return trackRADec


# /def


# ----------------------------------------------------------------------------


def merge_trailing_leading_tracks(trailing, leading):
    """Make a unified track from trailing, leading tracks.

    where trailing and leading are the standard output from get_track_RaDec

    Parameters
    ----------
    trailing, leading: (n, m) ndarrays
        streamdf tracks
        works with output from `get_track_RaDec`

    Returns
    -------
    track : (nt+nl, m) ndarray
        merged track of length nt+nl (trailing, leading)

    """
    # checking if the order should be reversed
    d1 = norm(trailing[0, :] - leading[0, :])
    d2 = norm(trailing[-1, :] - leading[0, :])
    d3 = norm(trailing[0, :] - leading[-1, :])
    d4 = norm(trailing[-1, :] - leading[-1, :])

    order = np.argmin([d1, d2, d3, d4])  # finding the minimizer

    if order == 0:
        trailing = trailing[::-1]
    elif order == 1:
        pass
    elif order == 2:
        trailing = trailing[::-1]
        leading = leading[::-1]
    elif order == 3:
        leading = leading[::-1]

    track = np.concatenate([trailing, leading])

    return track


# /def


# ----------------------------------------------------------------------------


def _setup_model_tracks(
    chain_index,
    potparams,
    chain_params,
    rng,
    setup_args=[],
    setup_kw={},
    interp_args=[],
    interp_kw={},
    # logger=_LOGFILE,
    # verbose=None,
    # _print=True,
):
    """Helper function for Setting up model tracks."""
    # logger.newsection(f"model {chain_index}:", div=".", print=_print)

    model = ObjDict(f"params {chain_index}")

    pot, sdf_t, sdf_l = setup_model(
        potparams,
        chain_params[chain_index],
        *setup_args,
        num_t=2000,
        **setup_kw,
    )  # FIXME, num_t doesn't work
    # logger.verbort(
    #     f"setup model",
    #     f"setup model:\n\targs: {setup_args}\n\tkw: {setup_kw}",
    #     verbose=verbose,
    #     print=_print,
    # )

    # get track
    trackRADec_t, trackRADec_l = get_track_RADec(sdf_t, sdf_l)
    # concatenating tracks, time-ordered (trailing -> prog -> leading)
    trackRADec = merge_trailing_leading_tracks(trackRADec_t, trackRADec_l)

    # logger.verbort(f"made tracks", verbose=verbose, print=_print)

    # interpolation
    ind = inRange(trackRADec[:, 0], trackRADec[:, 1], rng=rng)
    interp_ra_to_dec = interp1d(trackRADec[ind][:, 0], trackRADec[ind][:, 1])

    # logger.verbort(
    #     f"interpolated merged tracks",
    #     f"interpolated merged tracks:"
    #     "\n\tfunc: {interp_func}\n\targs: {interp_args}\n\tkw: {interp_kw}",
    #     verbose=verbose,
    #     print=_print,
    # )

    # storing
    model.chain_index = chain_index
    model.pot = pot
    model.chain_params = chain_params[chain_index]
    # model.sdf_t = sdf_t
    # model.sdf_l = sdf_l
    # model.trackRADec_t = trackRADec_t
    # model.trackRADec_l = trackRADec_l
    model.trackRADec = trackRADec
    model.interp_ra_to_dec = interp_ra_to_dec

    return model


# /def

# ----------------------------------------------------------------------------


def setup_models_in_chain(
    index,
    step=100,
    rng=[[210, 300], [-40, 20]],
    direc="../../data/pal5_mcmc",
    mcmc_args=[],
    mcmc_kw={},
    setup_args=[],
    setup_kw={},
    interp_args=[],
    interp_kw={},
    # extra
    parallelize=True,
    numcores=None,
    save=False,
    # logger=_LOGFILE,
    # verbose=None,
):
    """Setup Models in Chain.

    Parameters
    ----------
    index : int
        index of the chain
    step : int
        step size through the chain params
    rng : 2x2 list
        range over which to restrict the tracks for interpolation
        this means the data can only be calcuated so long as the dec
            of the track is a function of the track's ra
    direc : str
        mcmc directory
    mcmc_args : list, optional
        args for `read_mcmc_chain`
    mcmc_kw : dict, optional
        kwargs for `read_mcmc_chain`
    setup_args : list, optional
        args for `setup_model`
    setup_kw : dict, optional
        kwargs for `setup_model`
    interp_args : list, optional
        args for `interp_func`
    interp_kw : dict, optional
        kwargs for `interp_func`

    Returns
    -------
    chain : ObjDict
        container for
    .potparams: list
        list of potential parameters
    .chain_params: numpy.ndarray
        array of chain parameters, shape=(nsamples,nparams)=(nsamples,7)
        These parameters are: halo flattening c, Vc/_REFV0, distance/22 kpc,
        2 parameters related to Pal 5's proper motion, log(sigma_v/0.4 km/s),
        the final entry is the log likelihood of the sample
    .models : list
        elements are:

        model : ObjDict
        container for
        .pot: galpy Potential
            the potential
        .sdf_t: galpy.df.streamdf instance
            the trailing arm
        .sdf_l: galpy.df.streamdf instance
            the leading arm
        .trackRADec_t : (n, 2) array
            the track in ra, dec of the streamdf trailing arm
        .trackRADec_l : (n, 2) array
            the track in ra, dec of the streamdf leading arm
        .trackRADec : (2n, 2 array)
            combined track
        .track_interp : function
            interpolation function taking observed ra to streamdf dec
    """
    # logger.newsection(f"Setting up chain, model, track #{index}:\n", div="=")

    # read_mcmc_chain
    potparams, chain_params = read_mcmc_chain(
        index, *mcmc_args, direc=direc, **mcmc_kw
    )
    # logger.verbort(
    #     f"read mcmc chain #{index}",
    #     f"read mcmc chain #{index}:\n\targs: {mcmc_args}\n\tkw: {mcmc_kw}",
    #     verbose=verbose,
    # )

    # setting up models
    indices = list(range(0, len(chain_params), step))

    # logger.verbort(
    #     "setting up model & tracks",
    #     f"setting up model & tracks:\n\tusing chain_params {indices}",
    #     verbose=verbose,
    # )

    if parallelize:
        res = parallel_map(
            _setup_model_tracks,
            indices,
            func_args=[potparams, chain_params, rng],
            func_kws={
                "setup_args": [],
                "setup_kw": {},
                "interp_args": [],
                "interp_kw": {},
                # "logger": logger,
                # "verbose": None,
                # "_print": False,
            },
            numcores=numcores,
        )
        models = list(res)
    else:
        models = []
        for chain_index in indices:
            model = _setup_model_tracks(
                chain_index,
                potparams,
                chain_params,
                rng,
                setup_args=[],
                setup_kw={},
                interp_args=[],
                interp_kw={},
                # logger=logger,
                # verbose=None,
            )
            models.append(model)
        # /for

    # packaging results
    chain = ObjDict(
        f"everything related to chain {index}",
        index=index,
        potparams=potparams,
        chain_params=chain_params,
        models=models,
    )

    # logger.verbort(
    #     "packaging results",
    #     "packaging results:"
    #     "\n\t.potparams    : params for potential"
    #     "\n\t.chain_params : params for mcmc chain"
    #     "\n\t.pot          : the potential"
    #     # '\n\t.sdf_t        : streamdf trailing arm'
    #     # '\n\t.sdf_l        : streamdf leading arm'
    #     # '\n\t.trackRADec_t : trailing arm track'
    #     # '\n\t.trackRADec_l : leading arm track'
    #     "\n\t.trackRADec   : combined track"
    #     "\n\t.track_interp : interpolated track function",
    #     verbose=verbose,
    # )

    # saving
    if isinstance(save, str):
        fpath = save
        save = True
    else:
        fpath = f"chain{index}-step{step}.pkl"

    if save:  # bool
        with open(fpath, "wb") as file:
            pickle.dump(chain, file)
        # logger.verbort(
        #     "saved results", f"saved results at {fpath}", verbose=verbose
        # )

    return chain


# /def


# ----------------------------------------------------------------------------


def plot_chains(drct, nwalkers=12, ffmt=".dat"):
    """Plot MCMC Chains.

    Parameters
    ----------
    drct : str
        directory

    """
    drct = drct + "/" if not drct.endswith("/") else drct

    # getting number of chains
    files = os.listdir(drct)  # all files in directory
    chains = np.sort(
        [f for f in files if f.endswith(ffmt)]
    )  # get files with right file format.
    print(chains)

    npot = len(chains)
    ncol = 4
    nrow = int(np.ceil(npot / ncol))
    fig = plt.figure(figsize=(16, nrow * ncol))  # todo, dynamic figsize
    cmap = plt.get_cmap("plasma")

    for en, (ii, fn) in enumerate(zip(range(len(chains)), chains)):
        # fn = drct + "mwpot14-fitsigma-%i.dat" % ii
        data = np.loadtxt(drct + fn, comments="#", delimiter=",")
        plt.subplot(nrow, 4, en + 1)
        sdata = np.reshape(
            data[:, -1], (len(data[:, 5]) // nwalkers, nwalkers)
        )
        for jj in range(nwalkers):
            if ii % 4 == 0 and jj == 0:
                tylabel = r"$\ln \mathcal{L}$"
            else:
                tylabel = None
            if ii // 4 == nrow - 1 and jj == 0:
                txlabel = r"$\#\ \mathrm{of\ steps}$"
            else:
                txlabel = None
            bovy_plot.bovy_plot(
                list(range(len(sdata[:, jj]))),
                sdata[:, jj],
                "-",
                alpha=0.4,
                color=cmap(jj / 11.0),
                # yrange=[-40.0, -15.0],
                ylabel=tylabel,
                xlabel=txlabel,
                gcf=True,
            )
            bovy_plot.bovy_text(
                r"$\mathrm{Potential}\ %i$" % ii, size=17.0, top_left=True
            )
        nburn = determine_nburn(drct + fn) // nwalkers
        plt.axvline(nburn, lw=2.0, zorder=1, color="k")

    plt.tight_layout()

    return fig


# /def


###############################################################################
### END
