'''Try to recover structure links from the distance between documents.'''

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
import logging

import numpy as np
import networkx as nx
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score


def load_structure():
    structure_df = pd.read_csv(ARGS.preprocessed_data, usecols=[
                               'url', 'connected_to'])

    structure_graph = nx.Graph()

    # Load nodes
    structure_graph.add_nodes_from(structure_df['url'].values)

    # Load edges
    for _, row in structure_df.iterrows():
        from_url = row['url']
        connected_to = row['connected_to']
        # Don't consider null values
        if not pd.isnull(connected_to):
            for to_url in connected_to.split(','):
                # Don't consider connections which are not pages themselves
                if to_url in structure_graph:
                    structure_graph.add_edge(from_url, to_url)
    structure_graph.remove_edges_from(nx.selfloop_edges(structure_graph))

    return structure_graph


def build_dataset(structure, content):
    x, y = [], []
    for from_node in range(structure.shape[0]):
        for to_node in range(from_node):
            x.append([content[from_node, to_node]])
            y.append(structure[from_node, to_node])

    return x, y


# The execution starts here
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('preprocessed_data',
                        help='filename of the CSV containing the preprocessed scraped data')
    PARSER.add_argument('distances_matrix',
                        help='filename of the .npz representing the distance matrix')
    ARGS = PARSER.parse_args()

    STRUCTURE = load_structure()
    STRUCTURE_ADJ = np.array(nx.to_numpy_matrix(STRUCTURE, dtype=np.int32))
    np.fill_diagonal(STRUCTURE_ADJ, 0)
    CONTENT_ADJ = np.load(ARGS.distances_matrix).astype(float)

    X, Y = build_dataset(STRUCTURE_ADJ, CONTENT_ADJ)
    UNIQUE, COUNTS = np.unique(Y, return_counts=True)
    COUNTS = COUNTS / np.sum(COUNTS)
    print('Unbalancedness:', UNIQUE, COUNTS)

    CLF = LogisticRegression(penalty='none', fit_intercept=False)
    CLF.fit(X, Y)
    print('Threshold:', CLF.coef_)
    print('Intercept:', CLF.intercept_)

    PREDICTED = CLF.predict(X)
    print('Accuracy:', accuracy_score(Y, PREDICTED))
    print('F1 score:', f1_score(Y, PREDICTED))
    print('MCC:', matthews_corrcoef(Y, PREDICTED))

    TESTS = [[val] for val in np.linspace(0, 1, 20)]
    print(CLF.predict(TESTS))
