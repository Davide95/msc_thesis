import argparse
import logging

import numpy as np
import networkx as nx
import pandas as pd


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


def RDN(G, n_sub):
    '''It samples a subgraph from G by random selcting n_sub nodes based on the degree distribution.'''
    n_nodes = len(G.nodes())
    assert n_sub <= n_nodes and n_sub > 0

    degree = np.asarray([d for n, d in G.degree()])
    norm_degree = (1/np.sum(degree))*degree

    sampled = np.random.choice(
        np.arange(n_nodes), n_sub, replace=False, p=norm_degree)

    return sampled


def build_pos_dataset(structure, content, sampled):
    train = []
    test = []

    for from_node in range(structure.shape[0]):
        for to_node in range(from_node):
            if structure[from_node, to_node] == 1:
                content_val = content[from_node, to_node]
                if from_node in sampled and to_node in sampled:
                    train.append([content_val, 1])
                else:
                    test.append([content_val, 1])

    return np.asarray(train), np.asarray(test)


def build_neg_dataset(structure, content, ratio):
    train = []
    test = []
    for from_node in range(structure.shape[0]):
        for to_node in range(from_node):
            if structure[from_node, to_node] == 0:
                content_val = content[from_node, to_node]
                if np.random.binomial(1, ratio):
                    train.append([content_val, 0])
                else:
                    test.append([content_val, 0])

    return np.asarray(train), np.asarray(test)


    # The execution starts here
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('preprocessed_data',
                        help='filename of the CSV containing the preprocessed scraped data')
    PARSER.add_argument('distances_matrix',
                        help='filename of the .npz representing the distance matrix')
    PARSER.add_argument('train_size',
                        help='It represent the percentage of the dataset to include in the train split',
                        type=float, default=0.7)
    ARGS = PARSER.parse_args()

    STRUCTURE = load_structure()
    STRUCTURE_ADJ = np.array(nx.to_numpy_matrix(STRUCTURE, dtype=np.int32))
    np.fill_diagonal(STRUCTURE_ADJ, 0)
    CONTENT_ADJ = np.load(ARGS.distances_matrix).astype(int)

    TRAIN_NUM = int(len(STRUCTURE.nodes()) * ARGS.train_size)
    TRAIN_NODES = RDN(STRUCTURE, TRAIN_NUM)

    TRAIN_POS, TEST_POS = build_pos_dataset(STRUCTURE_ADJ, CONTENT_ADJ, TRAIN_NODES)
    TRAIN_POS_RATIO = TRAIN_POS.shape[0] / (TRAIN_POS.shape[0] + TEST_POS.shape[0])
    print('Train (positive labels)', TRAIN_POS_RATIO, '% of the dataset.')

    TRAIN_NEG, TEST_NEG = build_neg_dataset(STRUCTURE_ADJ, CONTENT_ADJ, TRAIN_POS_RATIO)
    TRAIN_NEG_RATIO = TRAIN_NEG.shape[0] / (TRAIN_NEG.shape[0] + TEST_NEG.shape[0])
    print('Train (negative labels)', TRAIN_NEG_RATIO, '% of the dataset.')
