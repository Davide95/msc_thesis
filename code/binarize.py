'''Given a distances matrix and a threshold, it returns a binarized network.'''

# Copyright (C) 2020 MaLGa ML4DS
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
from pathlib import Path

import numpy as np


def binarize():
    data = 1 - np.load(ARGS.filename)
    binarized = data >= ARGS.threshold
    np.fill_diagonal(binarized, False)

    return binarized


# The execution starts here
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('filename',
                        help='filename of the .npy containing the distance matrix')
    PARSER.add_argument(
        'threshold', type=float,
        help='Threshold found through the random_threshold step')
    ARGS = PARSER.parse_args()

    NETWORK = binarize()
    PATH = Path(ARGS.filename).stem + '-adj.npy'

    np.save(PATH, NETWORK)
    print('Output saved at:', PATH)
