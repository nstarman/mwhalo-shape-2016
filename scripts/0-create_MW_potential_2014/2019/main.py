# -*- coding: utf-8 -*-

# Docstring
"""Run the Full create_MW_potential_2014 set of scripts."""

__author__ = "Nathaniel Starkman"

__all__ = ["main"]


###############################################################################
# IMPORTS

# GENERAL
import argparse
import copy
from typing import Optional

# PROJECT-SPECIFIC

# fmt: off
import sys; sys.path.insert(0, "../../../")
# fmt: on
import pal5_constrain_mwhalo_shape.scripts.create_MW_potential_2014.script_util as su


###############################################################################
# CODE
###############################################################################


###############################################################################
# Command Line
###############################################################################


def make_parser(inheritable=False):
    """Make Parser.

    Parameters
    ----------
    inheritable: bool
        whether the parser can be inherited from (default False).
        if True, sets ``add_help=False`` and ``conflict_hander='resolve'``

    Returns
    -------
    parser: ArgumentParser

    """
    parser = argparse.ArgumentParser(
        description="Run full create_MW_potential_2014 set of scripts.",
        add_help=~inheritable,
        conflict_handler="resolve" if ~inheritable else "error",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/",
        type=str,
        help="output save folder",
        dest="opath",
    )
    parser.add_argument(
        "-f",
        "--figure",
        default="figures/",
        type=str,
        help="figure save folder",
        dest="fpath",
    )

    return parser


# /defs


# -------------------------------------------------------------------


def main(
    args: Optional[list] = None, opts: Optional[argparse.Namespace] = None
):
    """Run the Full create_MW_potential_2014 set of scripts.

    Parameters
    ----------
    args : list, optional
        an optional single argument that holds the sys.argv list,
        except for the script name (e.g., argv[1:])
    opts : Namespace, optional
        already parsed options.
        Will be ignored if args is not None.

    """
    if opts is not None and args is None:
        raise ValueError("TODO get args from parser")
    else:
        parser = make_parser()
        opts = parser.parse_args(args)

    # ---------------------------------

    from fit_mwpot14_script import main as run_fit_mwpot14
    from fit_mwpot_dblexp_script import main as run_fit_mwpot_dblexp
    from fit_potential_pal5_script import main as run_fit_potential_pal5
    from fit_potential_gd1_script import main as run_fit_potential_gd1
    from fit_potential_combo_pal5_gd1_script import (
        main as run_fit_potential_combo_pal5_gd1,
    )

    # ---------------------------------

    # 1) fit_mwpot14
    mw14opts = copy.copy(opts)
    mw14opts.fpath += "mwpot14/"
    run_fit_mwpot14(opts=mw14opts)

    # 2) fit_mwpot-dblexp
    dblopts = copy.copy(opts)
    dblopts.fpath += "mwpot_dblexp/"
    run_fit_mwpot_dblexp(opts=dblopts)

    # 3) Pal5
    pal5opts = copy.copy(opts)
    pal5opts.fpath += "pal5/"
    run_fit_potential_pal5(opts=pal5opts)

    # 4) GD1
    gd1opts = copy.copy(opts)
    gd1opts.fpath += "gd1/"
    run_fit_potential_gd1(opts=gd1opts)

    # 5) Combo Pal5 and GD1
    comboopts = copy.copy(opts)
    comboopts.fpath += "combo_pal5_gd1/"
    run_fit_potential_combo_pal5_gd1(opts=comboopts)

    # 6) plot force field
    su.plotForceField("figures/mwhalo-shapeforcefield.pdf")

    return


# /def


# ------------------------------------------------------------------------


if __name__ == "__main__":

    main(args=None, opts=None)

# /if


###############################################################################
# END
