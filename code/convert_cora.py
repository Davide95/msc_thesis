'''Convert CORA dataset to the CSV format that we use in the pipeline.'''

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

import os
import logging
import argparse

import networkx as nx
import pandas as pd


def load_structure():
    filename = os.path.join(ARGS.dataset_folder, 'cora.cites')
    edgelist = pd.read_csv(filename, sep='\t', header=None,
                           names=['target', 'source'])
    graph = nx.from_pandas_edgelist(edgelist, edge_attr=None)
    return graph


def load_content():
    filename = os.path.join(ARGS.dataset_folder, 'cora.content')
    feature_names = [f'w_{idx}' for idx in range(1433)]
    column_names = feature_names + ['subject']
    df = pd.read_csv(filename, sep='\t', names=column_names, header=None)
    df.drop(labels='subject', axis=1, inplace=True)
    return df


def convert(structure, content):
    url = []
    connected_to = []
    page_content = []

    for index, row in content.iterrows():
        url.append(str(index))
        page_content.append(','.join(map(str, row.values)))
        connected_to.append(','.join(map(str, structure.neighbors(index))))

    return pd.DataFrame(data={'url': url,
                              'connected_to': connected_to,
                              'content': page_content})


# The execution starts here
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('dataset_folder',
                        help='folder in which the already extracted dataset is')
    PARSER.add_argument('save_to',
                        help='filename of the output / the converted file')
    ARGS = PARSER.parse_args()

    STRUCTURE = load_structure()
    CONTENT = load_content()
    CONVERTED = convert(STRUCTURE, CONTENT)
    CONVERTED.to_csv(ARGS.save_to, index=False)
