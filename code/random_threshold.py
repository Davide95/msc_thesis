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
import glob

import numpy as np
import matplotlib.pyplot as plt


def random_threshold(filename):
    # Load data
    content = np.load(filename)
    content = 1 - content

    # Remove the diagonal
    np.fill_diagonal(content, -1)
    content = content.flatten()
    content = content[np.argwhere(content != -1)[:, 0]]

    # Compute the threshold
    vals, counts = np.unique(content, return_counts=True)
    counts = np.cumsum(counts)
    counts = 1 - counts / counts[-1]
    threshold_idx = np.argmax(counts <= ARGS.p_value)
    threshold = vals[threshold_idx]

    return threshold


# The execution starts here
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('pathname',
                        help='pathname of the CSVs containing the data')
    PARSER.add_argument(
        'p_value', type=float,
        help='Prior belief on how much our documents are truly indipdendent')
    PARSER.add_argument('--plot_thresholds', type=str, default=None,
                        help='Path where to plot the threshold values')
    ARGS = PARSER.parse_args()

    filenames = glob.glob(ARGS.pathname)
    print('Filenames extracted:', filenames)

    thresholds = []
    for filename in filenames:
        threshold = random_threshold(filename)
        thresholds.append(threshold)

    average = np.average(thresholds)
    std = np.std(thresholds)
    
    print('Threshold average:', average)
    print('Threshold standard deviation:', std)

    if ARGS.plot_thresholds is not None:
        plt.boxplot(thresholds)
        plt.ylabel('Hellinger distance')
        plt.title(f'Average: {average:.2g}, standard deviation: {std:.2g}')
        plt.savefig(ARGS.plot_thresholds)