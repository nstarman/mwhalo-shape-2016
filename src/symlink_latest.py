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

TODO, make this a general function

Routing Listings
----------------

"""

__author__ = "Nathaniel Starkman"
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
import os
import os.path

# CUSTOM

# PROJECT-SPECIFIC


###############################################################################
# PARAMETERS

_FOLDERS = ["0-create_MW_potential_2014", "1-palomar_5", "2-gd1"]
_SCRIPT_DIR = '../../'

###############################################################################
# CODE
###############################################################################


def sorted_folders(contents):
    # sort out files
    isfile = np.array([os.path.isfile(c) for c in contents])
    folders = contents[~isfile]

    # filter to date-start files
    allowed_start = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    folders = [f for f in folders if f[0] in allowed_start]

    return np.sort(folders)

# /def


# --------------------------------------------------------------------------

def make_symlinks(drct = _SCRIPT_DIR):

    for folder in _FOLDERS:

        old_dir = os.getcwd()
        os.chdir(drct + folder)

        contents = np.array(os.listdir('./'))
        folders = sorted_folders(contents)

        if len(folders) > 0:  # not empty
            try:
                os.rmdir('latest')
            except OSError:
                pass
            try:
                os.unlink('latest')
            except OSError:
                pass

            os.symlink('./' + folders[-1], './latest')

        os.chdir(old_dir)


###############################################################################
# Command Line
###############################################################################

if __name__ == '__main__':

    make_symlinks('../')

###############################################################################
# END
