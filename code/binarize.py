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

import networkx as nx
import numpy as np
import pandas as pd


def binarize():
    print('Loading distances matrix...')
    data = 1 - np.load(ARGS.distances_matrix)
    binarized = data >= ARGS.threshold

    print('Creation of the graph...')
    return nx.from_numpy_matrix(binarized)


def assign_labels(content):
    print('Loading scraped data...')
    structure = pd.read_csv(ARGS.scraped_data, usecols=['url'])

    print('Relabelling nodes...')
    labels = structure['url'].values
    indexes = list(structure.index)
    mapping = dict(zip(indexes, labels))
    nx.relabel_nodes(content, mapping, copy=False)


# The execution starts here
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('distances_matrix',
                        help='filename of the .npy containing the distance matrix')
    PARSER.add_argument('scraped_data',
                        help='filename of the CSV containing the scraped data')
    PARSER.add_argument(
        'threshold', type=float,
        help='Threshold found through the random_threshold step')
    ARGS = PARSER.parse_args()

    NETWORK = binarize()
    assign_labels(NETWORK)

    print('Saving data...')
    nx.write_gml(NETWORK, Path(ARGS.distances_matrix).stem + '.gml')
