# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#
# TITLE   :
# AUTHOR  :
# PROJECT :
#
# ----------------------------------------------------------------------------

# Docstring
"""**DOCSTRING**.

description

Routine Listings
----------------

"""

__author__ = ""
# __copyright__ = "Copyright 2019, "
# __credits__ = [""]
# __license__ = "MIT"
# __version__ = "0.0.0"
# __maintainer__ = ""
# __email__ = ""
# __status__ = "Production"

# __all__ = [
#     ""
# ]


###############################################################################
# IMPORTS

# GENERAL

import numpy as np

from galpy import potential
from galpy.util import bovy_coords

# CUSTOM

# PROJECT-SPECIFIC


###############################################################################
# PARAMETERS


_TKOP = np.array(
    [
        [-0.4776303088, -0.1738432154, 0.8611897727],
        [0.510844589, -0.8524449229, 0.111245042],
        [0.7147776536, 0.4930681392, 0.4959603976],
    ]
)


###############################################################################
# CODE
###############################################################################


def force_gd1(pot, ro, vo):
    """Return the force at GD-1.

    Parameters
    ----------
    pot: Potential
    ro, vo: float

    Return
    ------
    force: tuple
        [fx, fy, fz]

    """
    # Just use R=12.5 kpc, Z= 6.675 kpc, phi=0
    R1 = 12.5
    Z1 = 6.675
    p1 = 0.0

    return (
        potential.evaluateRforces(
            pot, R1 / ro, Z1 / ro, phi=p1, use_physical=True, ro=ro, vo=vo
        ),
        potential.evaluatezforces(
            pot, R1 / ro, Z1 / ro, phi=p1, use_physical=True, ro=ro, vo=vo
        ),
        potential.evaluatephiforces(
            pot, R1 / ro, Z1 / ro, phi=p1, use_physical=True, ro=ro, vo=vo
        ),
    )


# /def


# ---------------------------------------------------------------------


@bovy_coords.scalarDecorator
@bovy_coords.degreeDecorator([0, 1], [0, 1])
def lb_to_phi12(l, b, degree=False):
    """
    NAME:
       lb_to_phi12
    PURPOSE:
       Transform Galactic coordinates (l,b) to (phi1,phi2)
    INPUT:
       l - Galactic longitude (rad or degree)
       b - Galactic latitude (rad or degree)
       degree= (False) if True, input and output are in degrees
    OUTPUT:
       (phi1,phi2) for scalar input
       [:,2] array for vector input
    HISTORY:
        2014-11-04 - Written - Bovy (IAS)
    """
    # First convert to ra and dec
    radec = bovy_coords.lb_to_radec(l, b)
    ra = radec[:, 0]
    dec = radec[:, 1]
    XYZ = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec),])
    phiXYZ = np.dot(_TKOP, XYZ)
    phi2 = np.arcsin(phiXYZ[2])
    phi1 = np.arctan2(phiXYZ[1], phiXYZ[0])
    phi1[phi1 < 0.0] += 2.0 * np.pi
    return np.array([phi1, phi2]).T


# /def


# ---------------------------------------------------------------------


@bovy_coords.scalarDecorator
@bovy_coords.degreeDecorator([0, 1], [0, 1])
def phi12_to_lb(phi1, phi2, degree=False):
    """
    NAME:
       phi12_to_lb
    PURPOSE:
       Transform (phi1,phi2) to Galactic coordinates (l,b)
    INPUT:
       phi1 - phi longitude (rad or degree)
       phi2 - phi latitude (rad or degree)
       degree= (False) if True, input and output are in degrees
    OUTPUT:
       (l,b) for scalar input
       [:,2] array for vector input
    HISTORY:
        2014-11-04 - Written - Bovy (IAS)
    """
    # Convert phi1,phi2 to l,b coordinates
    phiXYZ = np.array(
        [np.cos(phi2) * np.cos(phi1), np.cos(phi2) * np.sin(phi1), np.sin(phi2),]
    )
    eqXYZ = np.dot(_TKOP.T, phiXYZ)
    # Get ra dec
    dec = np.arcsin(eqXYZ[2])
    ra = np.arctan2(eqXYZ[1], eqXYZ[0])
    ra[ra < 0.0] += 2.0 * np.pi
    # Now convert to l,b
    return bovy_coords.radec_to_lb(ra, dec)


# /def


# ---------------------------------------------------------------------


@bovy_coords.scalarDecorator
@bovy_coords.degreeDecorator([2, 3], [])
def pmllpmbb_to_pmphi12(pmll, pmbb, l, b, degree=False):
    """
    NAME:
       pmllpmbb_to_pmphi12
    PURPOSE:
       Transform proper motions in Galactic coordinates (l,b) to (phi1,phi2)
    INPUT:
       pmll - proper motion Galactic longitude (rad or degree); contains xcosb
       pmbb - Galactic latitude (rad or degree)
       l - Galactic longitude (rad or degree)
       b - Galactic latitude (rad or degree)
       degree= (False) if True, input (l,b) are in degrees
    OUTPUT:
       (pmphi1,pmphi2) for scalar input
       [:,2] array for vector input
    HISTORY:
        2014-11-04 - Written - Bovy (IAS)
    """
    # First go to ra and dec
    radec = bovy_coords.lb_to_radec(l, b)
    ra = radec[:, 0]
    dec = radec[:, 1]
    pmradec = bovy_coords.pmllpmbb_to_pmrapmdec(pmll, pmbb, l, b, degree=False)
    pmra = pmradec[:, 0]
    pmdec = pmradec[:, 1]
    # Now transform ra,dec pm to phi1,phi2
    phi12 = lb_to_phi12(l, b, degree=False)
    phi1 = phi12[:, 0]
    phi2 = phi12[:, 1]
    # Build A and Aphi matrices
    A = np.zeros((3, 3, len(ra)))
    A[0, 0] = np.cos(ra) * np.cos(dec)
    A[0, 1] = -np.sin(ra)
    A[0, 2] = -np.cos(ra) * np.sin(dec)
    A[1, 0] = np.sin(ra) * np.cos(dec)
    A[1, 1] = np.cos(ra)
    A[1, 2] = -np.sin(ra) * np.sin(dec)
    A[2, 0] = np.sin(dec)
    A[2, 1] = 0.0
    A[2, 2] = np.cos(dec)
    AphiInv = np.zeros((3, 3, len(ra)))
    AphiInv[0, 0] = np.cos(phi1) * np.cos(phi2)
    AphiInv[0, 1] = np.cos(phi2) * np.sin(phi1)
    AphiInv[0, 2] = np.sin(phi2)
    AphiInv[1, 0] = -np.sin(phi1)
    AphiInv[1, 1] = np.cos(phi1)
    AphiInv[1, 2] = 0.0
    AphiInv[2, 0] = -np.cos(phi1) * np.sin(phi2)
    AphiInv[2, 1] = -np.sin(phi1) * np.sin(phi2)
    AphiInv[2, 2] = np.cos(phi2)
    TA = np.dot(_TKOP, np.swapaxes(A, 0, 1))
    # Got lazy...
    trans = np.zeros((2, 2, len(ra)))
    for ii in range(len(ra)):
        trans[:, :, ii] = np.dot(AphiInv[:, :, ii], TA[:, :, ii])[1:, 1:]
    return (trans * np.array([[pmra, pmdec], [pmra, pmdec]])).sum(1).T


# /def


# ---------------------------------------------------------------------


@bovy_coords.scalarDecorator
@bovy_coords.degreeDecorator([2, 3], [])
def pmphi12_to_pmllpmbb(pmphi1, pmphi2, phi1, phi2, degree=False):
    """
    NAME:
       pmllpmbb_to_pmphi12
    PURPOSE:
       Transform proper motions in (phi1,phi2) to Galactic coordinates (l,b)
    INPUT:
       pmphi1 - proper motion Galactic longitude (rad or degree); contains xcosphi2
       pmphi2 - Galactic latitude (rad or degree)
       phi1 - phi longitude (rad or degree)
       phi2 - phi latitude (rad or degree)
       degree= (False) if True, input (phi1,phi2) are in degrees
    OUTPUT:
       (pmll,pmbb) for scalar input
       [:,2] array for vector input
    HISTORY:
        2014-11-04 - Written - Bovy (IAS)
    """
    # First go from phi12 to ra and dec
    lb = phi12_to_lb(phi1, phi2)
    radec = bovy_coords.lb_to_radec(lb[:, 0], lb[:, 1])
    ra = radec[:, 0]
    dec = radec[:, 1]
    # Build A and Aphi matrices
    AInv = np.zeros((3, 3, len(ra)))
    AInv[0, 0] = np.cos(ra) * np.cos(dec)
    AInv[0, 1] = np.sin(ra) * np.cos(dec)
    AInv[0, 2] = np.sin(dec)
    AInv[1, 0] = -np.sin(ra)
    AInv[1, 1] = np.cos(ra)
    AInv[1, 2] = 0.0
    AInv[2, 0] = -np.cos(ra) * np.sin(dec)
    AInv[2, 1] = -np.sin(ra) * np.sin(dec)
    AInv[2, 2] = np.cos(dec)
    Aphi = np.zeros((3, 3, len(ra)))
    Aphi[0, 0] = np.cos(phi1) * np.cos(phi2)
    Aphi[0, 1] = -np.sin(phi1)
    Aphi[0, 2] = -np.cos(phi1) * np.sin(phi2)
    Aphi[1, 0] = np.sin(phi1) * np.cos(phi2)
    Aphi[1, 1] = np.cos(phi1)
    Aphi[1, 2] = -np.sin(phi1) * np.sin(phi2)
    Aphi[2, 0] = np.sin(phi2)
    Aphi[2, 1] = 0.0
    Aphi[2, 2] = np.cos(phi2)
    TAphi = np.dot(_TKOP.T, np.swapaxes(Aphi, 0, 1))
    # Got lazy...
    trans = np.zeros((2, 2, len(ra)))
    for ii in range(len(ra)):
        trans[:, :, ii] = np.dot(AInv[:, :, ii], TAphi[:, :, ii])[1:, 1:]
    pmradec = (trans * np.array([[pmphi1, pmphi2], [pmphi1, pmphi2]])).sum(1).T
    pmra = pmradec[:, 0]
    pmdec = pmradec[:, 1]
    # Now convert to pmll
    return bovy_coords.pmrapmdec_to_pmllpmbb(pmra, pmdec, ra, dec)


# /def


# ---------------------------------------------------------------------


def convert_track_lb_to_phi12(track):
    """track = _interpolatedObsTrackLB"""
    phi12 = lb_to_phi12(track[:, 0], track[:, 1], degree=True)
    phi12[phi12[:, 0] > 180.0, 0] -= 360.0
    pmphi12 = pmllpmbb_to_pmphi12(
        track[:, 4], track[:, 5], track[:, 0], track[:, 1], degree=True
    )
    out = np.empty_like(track)
    out[:, :2] = phi12
    out[:, 2] = track[:, 2]
    out[:, 3] = track[:, 3]
    out[:, 4:] = pmphi12
    return out


# /def


# ---------------------------------------------------------------------


def phi12_to_lb_6d(phi1, phi2, dist, pmphi1, pmphi2, vlos):
    l, b = phi12_to_lb(phi1, phi2, degree=True)
    pmll, pmbb = pmphi12_to_pmllpmbb(pmphi1, pmphi2, phi1, phi2, degree=True)
    return [l, b, dist, pmll, pmbb, vlos]


# /def


###############################################################################
# END
