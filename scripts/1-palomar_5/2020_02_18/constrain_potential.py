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

Routing Listings
----------------

"""

__author__ = "Nathaniel Starkman & Jo Bovy"
__maintainer__ = "Nathaniel Starkman"

# __all__ = [
#     ""
# ]


###############################################################################
# IMPORTS

# GENERAL

# CUSTOM

# PROJECT-SPECIFIC
# fmt: off
import sys; sys.path.insert(0, '../../../')
# fmt: on
from src import MWPotential2014Likelihood
from src import mcmc_util


###############################################################################
# PARAMETERS


###############################################################################
# CODE
###############################################################################

# class ClassName(object):
#     """Docstring for ClassName."""

#     def __init__(self, arg):
#         """Initialize class."""
#         super().__init__()
#         self.arg = arg
# # /class


# # --------------------------------------------------------------------------

# def function():
#     """Docstring."""
#     pass
# # /def


###############################################################################
# Command Line
###############################################################################


fig = mcmc_util.plot_chains('output/fitsigma/')
fig.savefig('figures/chains.pdf')


###############################################################################
# END
