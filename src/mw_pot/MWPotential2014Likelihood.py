# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   : MWPotential2014Likelihood
# PROJECT : Pal 5 update MW pot constraints
#
# ----------------------------------------------------------------------------

# Docstring
"""Milky Way Potential (2014 version) Likelihood.

Routing Listings
----------------
like_func
pdf_func
setup_potential
force_pal5
force_gd1
mass60
bulge_dispersion
visible_dens
logprior_dlnvcdlnr
plotRotcurve
plotKz
plotTerm
plotPot
plotDens
readClemens
readMcClureGriffiths
readMcClureGriffiths16
calc_corr
binlbins

References
----------
https://github.com/jobovy/mwhalo-shape-2016

"""

__author__ = "Jo Bovy"
__copyright__ = "Copyright 2016, 2020, "
__maintainer__ = "Nathaniel Starkman"

# __all__ = [
#     ""
# ]


###############################################################################
# IMPORTS

# GENERAL

import functools
from typing import Sequence, Tuple, Union, Optional

import numpy as np
from scipy import optimize
import emcee
from tqdm import tqdm

import astropy.units as u

from galpy import potential
from galpy.potential import Potential
from galpy.util import bovy_conversion

import matplotlib.pyplot as plt


# CUSTOM

import bovy_mcmc  # TODO not need


# PROJECT-SPECIFIC

from .data import (  # import here for backward compatibility
    readBovyRix13kzdata,
    readClemens,
    readMcClureGriffiths07,
    readMcClureGriffiths16,
)
from .utils import (
    REFR0,
    REFV0,
    _get_data_and_make_funcargs,
    mass60,
    bulge_dispersion,
    visible_dens,
)

from ..streams.pal5.pal5_util import force_pal5
from ..streams.gd1.gd1_util import force_gd1

from . import plot


###############################################################################
# PARAMETERS

_REFR0, _REFV0 = REFR0, REFV0  # TODO deprecate

PotentialType = Union[Potential, Sequence[Potential]]


###############################################################################
# CODE
###############################################################################


def like_func(
    params: Sequence,
    c: float,
    surfrs: list,
    kzs,
    kzerrs,
    termdata,
    termsigma,
    fitc,
    fitvoro,
    dblexp,
    addpal5,
    addgd1,
    ro: float,
    vo: float,
    addgas: bool,
):
    """Likelihood Function.

    Parameters
    ----------
    params: list
    c: float
    surfrs: list
    kzs
    kzerrs
    termdata
    termsigma
    fitc
    fitvoro
    dblexp
    addpal5
    addgd1
    ro: float
    vo: float
    addgas: bool

    Returns
    -------
    float

    """
    # --------------------------------------------------------------------
    # Check ranges

    if params[0] < 0.0 or params[0] > 1.0:
        return np.finfo(np.dtype(np.float64)).max
    elif params[1] < 0.0 or params[1] > 1.0:
        return np.finfo(np.dtype(np.float64)).max
    elif (1.0 - params[0] - params[1]) < 0.0 or (
        1.0 - params[0] - params[1]
    ) > 1.0:
        return np.finfo(np.dtype(np.float64)).max
    elif params[2] < np.log(1.0 / REFR0) or params[2] > np.log(8.0 / REFR0):
        return np.finfo(np.dtype(np.float64)).max
    elif params[3] < np.log(0.05 / REFR0) or params[3] > np.log(1.0 / REFR0):
        return np.finfo(np.dtype(np.float64)).max
    elif fitvoro and (params[7] <= 150.0 / REFV0 or params[7] > 290.0 / REFV0):
        return np.finfo(np.dtype(np.float64)).max
    elif fitvoro and (params[8] <= 7.0 / REFR0 or params[8] > 9.4 / REFR0):
        return np.finfo(np.dtype(np.float64)).max
    elif fitc and (
        params[7 + 2 * fitvoro] <= 0.0 or params[7 + 2 * fitvoro] > 4.0
    ):
        return np.finfo(np.dtype(np.float64)).max

    # --------------------------------------------------------------------

    if fitvoro:
        ro, vo = REFR0 * params[8], REFV0 * params[7]

    # Setup potential
    pot = setup_potential(
        params, c, fitc, dblexp, ro, vo, fitvoro=fitvoro, addgas=addgas
    )

    # Calculate model surface density at surfrs
    modelkzs = np.empty_like(surfrs)
    for ii in range(len(surfrs)):
        modelkzs[ii] = -potential.evaluatezforces(
            pot, (ro - 8.0 + surfrs[ii]) / ro, 1.1 / ro, phi=0.0
        ) * bovy_conversion.force_in_2piGmsolpc2(vo, ro)
    out = 0.5 * np.sum((kzs - modelkzs) ** 2.0 / kzerrs ** 2.0)

    # Add terminal velocities
    vrsun = params[5]
    vtsun = params[6]
    cl_glon, cl_vterm, cl_corr, mc_glon, mc_vterm, mc_corr = termdata

    # Calculate terminal velocities at data glon
    cl_vterm_model = np.zeros_like(cl_vterm)
    for ii in range(len(cl_glon)):
        cl_vterm_model[ii] = potential.vterm(pot, cl_glon[ii])
    cl_vterm_model += vrsun * np.cos(cl_glon / 180.0 * np.pi) - vtsun * np.sin(
        cl_glon / 180.0 * np.pi
    )
    mc_vterm_model = np.zeros_like(mc_vterm)
    for ii in range(len(mc_glon)):
        mc_vterm_model[ii] = potential.vterm(pot, mc_glon[ii])
    mc_vterm_model += vrsun * np.cos(mc_glon / 180.0 * np.pi) - vtsun * np.sin(
        mc_glon / 180.0 * np.pi
    )
    cl_dvterm = (cl_vterm - cl_vterm_model * vo) / termsigma
    mc_dvterm = (mc_vterm - mc_vterm_model * vo) / termsigma
    out += 0.5 * np.sum(cl_dvterm * np.dot(cl_corr, cl_dvterm))
    out += 0.5 * np.sum(mc_dvterm * np.dot(mc_corr, mc_dvterm))

    # Rotation curve constraint
    out -= logprior_dlnvcdlnr(potential.dvcircdR(pot, 1.0, phi=0.0))

    # K dwarfs, Kz
    out += (
        0.5
        * (
            -potential.evaluatezforces(pot, 1.0, 1.1 / ro, phi=0.0)
            * bovy_conversion.force_in_2piGmsolpc2(vo, ro)
            - 67.0
        )
        ** 2.0
        / 36.0
    )
    # K dwarfs, visible
    out += 0.5 * (visible_dens(pot, ro, vo) - 55.0) ** 2.0 / 25.0
    # Local density prior
    localdens = potential.evaluateDensities(
        pot, 1.0, 0.0, phi=0.0
    ) * bovy_conversion.dens_in_msolpc3(vo, ro)
    out += 0.5 * (localdens - 0.102) ** 2.0 / 0.01 ** 2.0
    # Bulge velocity dispersion
    out += 0.5 * (bulge_dispersion(pot, ro, vo) - 117.0) ** 2.0 / 225.0
    # Mass at 60 kpc
    out += 0.5 * (mass60(pot, ro, vo) - 4.0) ** 2.0 / 0.7 ** 2.0

    # Pal5
    if addpal5:
        # q = 0.94 +/- 0.05 + add'l
        fp5 = force_pal5(pot, 23.46, ro, vo)
        out += (
            0.5 * (np.sqrt(2.0 * fp5[0] / fp5[1]) - 0.94) ** 2.0 / 0.05 ** 2.0
        )
        out += (
            0.5
            * (0.94 ** 2.0 * (fp5[0] + 0.8) + 2.0 * (fp5[1] + 1.82) + 0.2)
            ** 2.0
            / 0.6 ** 2.0
        )

    # GD-1
    if addgd1:
        # q = 0.95 +/- 0.04 + add'l
        fg1 = force_gd1(pot, ro, vo)
        out += (
            0.5
            * (np.sqrt(6.675 / 12.5 * fg1[0] / fg1[1]) - 0.95) ** 2.0
            / 0.04 ** 2.0
        )
        out += (
            0.5
            * (
                0.95 ** 2.0 * (fg1[0] + 2.51)
                + 6.675 / 12.5 * (fg1[1] + 1.47)
                + 0.05
            )
            ** 2.0
            / 0.3 ** 2.0
        )

    # vc and ro measurements: vc=218 +/- 10 km/s, ro= 8.1 +/- 0.1 kpc
    out += (vo - 218.0) ** 2.0 / 200.0 + (ro - 8.1) ** 2.0 / 0.02

    if np.isnan(out):
        return np.finfo(np.dtype(np.float64)).max
    else:
        return out


# /def


# --------------------------------------------------------------------------


def pdf_func(params: Sequence, *args) -> float:
    """PDF function.

    the negative likelihood

    Parameters
    ----------
    params: list
    c: float
    surfrs: list
    kzs
    kzerrs
    termdata
    termsigma
    fitc
    fitvoro
    dblexp
    addpal5
    addgd1
    ro: float
    vo: float
    addgas: bool

    Returns
    -------
    float

    """
    return -like_func(params, *args)


# /def


# --------------------------------------------------------------------------


def setup_potential(
    params: Sequence,
    c: float,
    fitc: bool,
    dblexp: bool,
    ro: float = REFR0,
    vo: float = REFV0,
    fitvoro: bool = False,
    b: float = 1.0,
    pa: float = 0.0,
    addgas: bool = False,
) -> PotentialType:
    """Set up potential.

    Parameters
    ----------
    params
    c
    fitc
    dblexp
    ro
    vo
    fitvoro: bool, optional
        default False
    b: float, optional
        default 1.0
    pa: float, optional
        Position Angle
        default 0.0
    addgas: bool, optional
        default False

    """
    pot: potential.Potential = [
        potential.PowerSphericalPotentialwCutoff(
            normalize=1.0 - params[0] - params[1], alpha=1.8, rc=1.9 / ro
        )
    ]

    if dblexp:
        if addgas:
            # add 13 Msun/pc^2
            gp = potential.DoubleExponentialDiskPotential(
                amp=0.03333
                * u.Msun
                / u.pc ** 3
                * np.exp(ro / 2.0 / np.exp(params[2]) / REFR0),
                hz=150.0 * u.pc,
                hr=2.0 * np.exp(params[2]) * REFR0 / ro,
                ro=ro,
                vo=vo,
            )
            gp.turn_physical_off()
            gprf = gp.Rforce(1.0, 0.0)
            dpf = params[0] + gprf
            if dpf < 0.0:
                dpf = 0.0
            pot.append(
                potential.DoubleExponentialDiskPotential(
                    normalize=dpf,
                    hr=np.exp(params[2]) * REFR0 / ro,
                    hz=np.exp(params[3]) * REFR0 / ro,
                )
            )
        else:
            pot.append(
                potential.DoubleExponentialDiskPotential(
                    normalize=params[0],
                    hr=np.exp(params[2]) * REFR0 / ro,
                    hz=np.exp(params[3]) * REFR0 / ro,
                )
            )
    else:
        pot.append(
            potential.MiyamotoNagaiPotential(
                normalize=params[0],
                a=np.exp(params[2]) * REFR0 / ro,
                b=np.exp(params[3]) * REFR0 / ro,
            )
        )

    if fitc:
        pot.append(
            potential.TriaxialNFWPotential(
                normalize=params[1],
                a=np.exp(params[4]) * REFR0 / ro,
                c=params[7 + 2 * fitvoro],
                b=b,
                pa=pa,
            )
        )
    else:
        pot.append(
            potential.TriaxialNFWPotential(
                normalize=params[1],
                a=np.exp(params[4]) * REFR0 / ro,
                c=c,
                b=b,
                pa=pa,
            )
        )

    if addgas:
        pot.append(gp)  # make sure it's the last

    return pot


# /def


# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def fit(
    fitc: bool = False,
    ro: float = REFR0,
    vo: float = REFV0,
    fitvoro: bool = False,
    c: float = 1.0,
    dblexp: bool = False,
    plots: bool = True,
    addpal5: bool = False,
    addgd1: bool = False,
    mc16: bool = False,
    addgas: bool = False,
) -> Tuple[Sequence, float]:
    """Perform a Fit of the potential to the data.

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
    (
        (surfrs, kzs, kzerrs),
        termdata,
        termdata_mc16,
        funcargs,
    ) = _get_data_and_make_funcargs(
        fitc, ro, vo, fitvoro, c, dblexp, addpal5, addgd1, mc16, addgas
    )

    # ---------------------

    init_params: list = [
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

    params: Tuple = optimize.fmin_powell(
        like_func, init_params, args=funcargs, disp=False,
    )

    ln_like: float = like_func(params, *funcargs)

    # ----------------

    if plots:

        if fitvoro:
            ro, vo = REFR0 * params[8], REFV0 * params[7]

        pot = setup_potential(params, c, fitc, dblexp, ro, vo, fitvoro=fitvoro)

        fig = plt.Figure()
        plt.subplot(1, 3, 1)
        plot.plotRotcurve(pot)
        plt.subplot(1, 3, 2)
        plot.plotKz(pot, surfrs, kzs, kzerrs, ro, vo)
        plt.subplot(1, 3, 3)
        if mc16:
            plot.plotTerm(pot, termdata_mc16, ro, vo)
        else:
            plot.plotTerm(pot, termdata, ro, vo)

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

    # ----------------

    return params, ln_like


# /def


# --------------------------------------------------------------------------


def sample(
    nsamples: int = 1000,
    params: Optional[Sequence] = None,
    fitc: bool = False,
    ro: float = REFR0,
    vo: float = REFV0,
    fitvoro: bool = False,
    c: float = 1.0,
    dblexp: bool = False,
    addpal5: bool = False,
    addgd1: bool = False,
    plots: bool = True,
    mc16: bool = False,
    addgas: bool = False,
    _use_emcee: bool = True,
) -> Sequence:
    """Sample from potential.

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
    (
        (surfrs, kzs, kzerrs),
        termdata,
        termdata_mc16,
        funcargs,
    ) = _get_data_and_make_funcargs(
        fitc, ro, vo, fitvoro, c, dblexp, addpal5, addgd1, mc16, addgas
    )

    # ---------------------

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

    samples = bovy_mcmc.markovpy(  # TODO Deprecated
        params,
        0.2,
        lambda x: -like_func(x, *funcargs),
        (),
        isDomainFinite=[[False, False] for ii in range(len(params))],
        domain=[[0.0, 0.0] for ii in range(len(params))],
        nsamples=nsamples,
        nwalkers=2 * len(params),
        _use_emcee=_use_emcee,
    )
    samples = np.array(samples).T

    if plots:
        plot.plot_samples(samples, fitc, fitvoro, ro=ro, vo=vo)

    return samples


# /def


# --------------------------------------------------------------------------


def sample_multi(
    nsamples: int = 1000,
    params: Sequence = None,
    fitc: bool = False,
    ro: float = REFR0,
    vo: float = REFV0,
    fitvoro: bool = False,
    c: float = 1.0,
    dblexp: bool = False,
    addpal5: bool = False,
    addgd1: bool = False,
    plots: bool = True,
    mc16: bool = False,
    addgas: bool = False,
) -> Sequence:
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
    (
        (surfrs, kzs, kzerrs),
        termdata,
        termdata_mc16,
        funcargs,
    ) = _get_data_and_make_funcargs(
        fitc, ro, vo, fitvoro, c, dblexp, addpal5, addgd1, mc16, addgas
    )

    # ---------------------

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

    nwalkers = 2 * len(params)
    nn = 0
    all_start_params = np.zeros((nwalkers, len(params)))
    start_lnprob0 = np.zeros(nwalkers)
    step = 0.05 * np.ones(len(params))

    while nn < nwalkers:
        all_start_params[nn] = (
            params + np.random.normal(size=len(params)) * step
        )
        start_lnprob0[nn] = pdf_func(all_start_params[nn], *funcargs)
        if start_lnprob0[nn] > -1000000.0:
            nn += 1

    sampler = emcee.EnsembleSampler(
        nwalkers, len(params), pdf_func, args=funcargs, threads=len(params),
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
        plot.plot_samples(samples, fitc, fitvoro, ro=ro, vo=vo)

    return samples


# /def


# --------------------------------------------------------------------------


def logprior_dlnvcdlnr(dlnvcdlnr: float) -> float:
    """Log Prior dlnvcdlnr.

    Parameters
    ----------
    dlnvcdlnr : float

    Returns
    -------
    float

    """
    sb = 0.04
    if dlnvcdlnr > sb or dlnvcdlnr < -0.5:
        return -np.finfo(np.dtype(np.float64)).max
    return np.log((sb - dlnvcdlnr) / sb) - (sb - dlnvcdlnr) / sb


# /def


##############################################################################
# END
