# -*- coding: utf-8 -*-
# see LICENSE.rst

# ----------------------------------------------------------------------------
#
# TITLE   : Kuepper et al. (2015)
# AUTHOR  : Jo Bovy and Nathaniel Starkman
# PROJECT : Pal 5 update MW potential constraints
#
# ----------------------------------------------------------------------------

"""Functions related to Kuepper et al. (2015).

Routine Listings
----------------
sample_kuepper_flatteningforce_post
plot_kuepper_samples
force_pal5_kuepper
kuepper_flattening_post
kuepper_flatteningforce_post
sample_kuepper_flatteningforce_post

"""

__author__ = ["Nathaniel Starkman", "Jo Bovy"]
__maintainer__ = "Nathaniel Starkman"

__all__ = [
    "sample_kuepper_flatteningforce_post",
    "plot_kuepper_samples",
    "force_pal5_kuepper",
    "kuepper_flattening_post",
    "kuepper_flatteningforce_post",
    "sample_kuepper_flatteningforce_post",
]


##############################################################################
# IMPORTS

# GENERAL

from typing import Sequence, List, Tuple

import numpy as np

import astropy.units as u

import corner

from galpy import potential
from galpy.potential import Potential


# CUSTOM

import bovy_mcmc


###############################################################################
# PARAMETERS


###############################################################################
# CODE
###############################################################################


def plot_kuepper_samples(samples: Sequence) -> None:
    """Plot Kuepper et al. (2015) Samples.

    Parameters
    ----------
    samples: ndarray

    """
    labels = [
        r"$M_{\mathrm{halo}} / (10^{12}\,M_\odot)$",
        r"$q_z$",
        r"$a / \mathrm{kpc}$",
    ]
    ranges = [(0.0, 4.0), (0.2, 1.8), (0.0, 100.0)]

    # fmt: off
    import pdb; pdb.set_trace()
    # fmt: on

    corner.corner(
        samples[[0, 2, 1]].T,
        quantiles=[0.16, 0.5, 0.84],
        labels=labels,
        show_titles=True,
        title_args={"fontsize": 12},
        range=ranges,
    )

    return


# /def


# -------------------------------------------------------------------


def setup_potential_kuepper(Mhalo: float, a: float) -> List[Potential]:
    """Set up Kuepper et al. (2015) Potential.

    Parameters
    ----------
    Mhalo: float
        mass / 10^12 Msun
    a: float
        scale length / kpc,

    Returns
    -------
    potential : list
        HernquistPotential + MiyamotoNagaiPotential + NFWPotential

    """
    pot: List[Potential] = [
        potential.HernquistPotential(amp=3.4e10 * u.Msun, a=0.7 * u.kpc),
        potential.MiyamotoNagaiPotential(
            amp=1e11 * u.Msun, a=6.5 * u.kpc, b=0.26 * u.kpc
        ),
        potential.NFWPotential(amp=Mhalo * 1e12 * u.Msun, a=a * u.kpc),
    ]

    return pot


# /def


# -------------------------------------------------------------------


def force_pal5_kuepper(pot: Sequence[Potential], qNFW: float) -> Tuple[float]:
    """Kuepper et al. (2015) force on Pal 5.

    Parameters
    ----------
    pot : Potential
    qNFW : float

    Returns
    -------
    FR : float
    FZ : float

    """
    R = 8.4 * u.kpc
    Z = 16.8 * u.kpc

    FR = pot[0].Rforce(R, Z) + pot[1].Rforce(R, Z) + pot[2].Rforce(R, Z / qNFW)
    FZ = pot[0].zforce(R, Z) + pot[1].zforce(R, Z) + pot[2].zforce(R, Z / qNFW)

    return FR, FZ


# /def


# -------------------------------------------------------------------


def kuepper_flattening_post(
    params: Sequence[float], qmean: float, qerr: float
) -> float:
    """A Kuepper et al. (2015) - flattened force posterior.

    Consists solely of the priors and a constraint on q_Phi.

    Parameters
    ----------
    params : list
        (3,) list. The Mhalo, a, qNFW
    qmean : float
    qerr : float

    Returns
    -------
    float

    """
    Mhalo, a, qNFW = params

    if Mhalo < 0.001 or Mhalo > 10.0:
        return -1e18
    elif a < 0.1 or a > 100.0:
        return -1e18
    elif qNFW < 0.2 or qNFW > 1.8:
        return -1e18

    pot: list = setup_potential_kuepper(Mhalo, a)
    vcpred = potential.vcirc(pot, 8.0 * u.kpc)

    if vcpred < 200.0 or vcpred > 280.0:
        return -1e18

    FR, FZ = force_pal5_kuepper(pot, qNFW)
    qpred = np.sqrt(2.0 * FR / FZ)

    post: float = -0.5 * np.square((qpred - qmean) / qerr)

    return post


# /def


# -------------------------------------------------------------------


def kuepper_flatteningforce_post(
    params: Sequence[float], qmean: float, qerr: float, frfz: float, frfzerr: float,
) -> float:
    """A Kuepper et al. (2015) - flattened force posterior.

    Consists solely of the priors, a constraint on q_Phi, and on FR+FZ.

    Parameters
    ----------
    params : list
        (3,) list. The Mhalo, a, qNFW
    qmean : float
    qerr : float
    frfz : float
    frfzerr : float

    Returns
    -------
    float

    """
    post: float = kuepper_flattening_post(params, qmean, qerr, frfz)

    if post == -1e18:
        return post

    else:

        Mhalo, a, qNFW = params
        pot = setup_potential_kuepper(Mhalo, a)
        FR, FZ = force_pal5_kuepper(pot, qNFW)
        frfzpred = (FR + 0.8) + (FZ + 1.83)

        return post - 0.5 * np.square((frfz - frfzpred) / frfzerr)

    # Mhalo, a, qNFW = params

    # if Mhalo < 0.001 or Mhalo > 10.0:
    #     return -1e18
    # elif a < 0.1 or a > 100.0:
    #     return -1e18
    # elif qNFW < 0.2 or qNFW > 1.8:
    #     return -1e18

    # pot = setup_potential_kuepper(Mhalo, a)
    # vcpred = potential.vcirc(pot, 8.0 * u.kpc)

    # if vcpred < 200.0 or vcpred > 280.0:
    #     return -1e18

    # FR, FZ = force_pal5_kuepper(pot, qNFW)
    # qpred = np.sqrt(2.0 * FR / FZ)
    # frfzpred = (FR + 0.8) + (FZ + 1.83)

    # return (
    #     -0.5 * (qpred - qmean) ** 2.0 / qerr ** 2.0
    #     - 0.5 * (frfz - frfzpred) ** 2.0 / frfzerr ** 2.0
    # )


# /def


# -------------------------------------------------------------------


def sample_kuepper_flattening_post(
    nsamples: int, qmean: float, qerr: float
) -> Sequence:
    """Sample Kuepper et al. (2015) flattening posterior.

    Parameters
    ----------
    nsamples : int
    qmean : float
    qerr : float

    Returns
    -------
    samples : ndarray

    """
    params = [1.58, 37.9, 0.95]
    funcargs = (qmean, qerr)

    samples = bovy_mcmc.markovpy(
        params,
        0.2,
        lambda x: kuepper_flattening_post(x, *funcargs),
        (),
        isDomainFinite=[[False, False] for ii in range(len(params))],
        domain=[[0.0, 0.0] for ii in range(len(params))],
        nsamples=nsamples,
        nwalkers=2 * len(params),
        _use_emcee=True,
    )

    samples = np.array(samples).T

    return samples


# /def


# -------------------------------------------------------------------


def sample_kuepper_flatteningforce_post(
    nsamples: int, qmean: float, qerr: float, frfz: float, frfzerr: float
) -> Sequence:
    """Sample Kuepper et al. (2015) Flattening Force Posterior.

    Parameters
    ----------
    nsamples : int
    qmean : float
    qerr : float
    frfz : float
    frfzerr : float

    Returns
    -------
    samples : ndarray

    """
    params = [1.58, 37.9, 0.95]
    funcargs = (qmean, qerr, frfz, frfzerr)

    samples = bovy_mcmc.markovpy(
        params,
        0.2,
        lambda x: kuepper_flatteningforce_post(x, *funcargs),
        (),
        isDomainFinite=[[False, False] for ii in range(len(params))],
        domain=[[0.0, 0.0] for ii in range(len(params))],
        nsamples=nsamples,
        nwalkers=2 * len(params),
    )

    samples = np.array(samples).T

    return samples


# /def


###############################################################################
# END
