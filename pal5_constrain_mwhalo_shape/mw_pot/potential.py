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
from .likelihood import pdf_func, like_func


###############################################################################
# PARAMETERS

PotentialType = Union[Potential, Sequence[Potential]]


###############################################################################
# CODE
###############################################################################


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

    PowerSphericalPotentialwCutoff
    MiyamotoNagaiPotential or DoubleExponentialDiskPotential
    TriaxialNFWPotential

    Parameters
    ----------
    params
    c
    fitc
    dblexp
        DoubleExponentialDiskPotential instead if MiyamotoNagaiPotential
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
                normalize=params[1], a=np.exp(params[4]) * REFR0 / ro, c=c, b=b, pa=pa,
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
    (kzdata, termdata, termdata_mc16, funcargs,) = _get_data_and_make_funcargs(
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
        like_func, init_params, args=tuple(funcargs), disp=False,
    )

    ln_like: float = like_func(params, *funcargs)

    # ----------------

    if plots:

        if fitvoro:
            ro, vo = REFR0 * params[8], REFV0 * params[7]

        pot = setup_potential(params, c, fitc, dblexp, ro, vo, fitvoro=fitvoro)

        if not isinstance(plots, str):
            savefig = (
                f"figures/fit-fitc_{fitc}-fitvoro_{fitvoro}-c_{c}-"
                f"dblexp_{dblexp}-addpal5_{addpal5}-addgd1_{addgd1}-"
                f"mc16_{mc16}-addgas_{addgas}.png"
            )
        else:
            savefig = plots

        if mc16:
            termdata = termdata_mc16

        plot.plotFit(
            pot=pot,
            kzdata=kzdata,
            termdata=termdata,
            ro=ro,
            vo=vo,
            suptitle=r"p: " + str(params) + r"  $\mathcal{L}$:" + str(ln_like),
            savefig=savefig,
        )

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

    samples = bovy_mcmc.markovpy(  # TODO Deprecate
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
        all_start_params[nn] = params + np.random.normal(size=len(params)) * step
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
            all_start_params, 1, log_prob0=start_lnprob0, rstate0=rstate0, store=False,
        )
        all_start_params = new_params
        start_lnprob0 = new_lnp
        rstate0 = new_rstate0

    for ii in tqdm(range(nsamples // nwalkers + 1)):  # burn-in
        new_params, new_lnp, new_rstate0 = sampler.run_mcmc(
            all_start_params, 1, log_prob0=start_lnprob0, rstate0=rstate0, store=False,
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


##############################################################################
# END
