'''Compute the random threshold from indipdendent samples.'''

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
import numpy as np


def random_threshold():
    # Load data
    content = np.load(ARGS.filename)
    content = 1 - content

    # Remove the diagonal
    np.fill_diagonal(content, -1)
    content = content.flatten()
    content = content[np.argwhere(content != -1)[:, 0]]

    # Compute the threshold
    vals, counts = np.unique(content, return_counts=True)
    counts = np.cumsum(counts)
    counts = 1 - np.cumsum(counts)/np.sum(counts)
    threshold_idx = np.argmax(counts <= ARGS.p_value)
    threshold = vals[threshold_idx]

    print('Threshold:', threshold)


# The execution starts here
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('filename',
                        help='filename of the CSV containing the data')
    PARSER.add_argument(
        'p_value', type=float,
        help='Prior belief on how much our documents are truly indipdendent')
    ARGS = PARSER.parse_args()

    random_threshold()
