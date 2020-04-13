'''Compute the similarity graph.'''

# Copyright (C) 2020  Davide Riva <driva95@protonmail.com>
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
import gc
import logging
import multiprocessing
import os
import time
from pathlib import Path
import math
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from gensim.matutils import Sparse2Corpus, corpus2dense
from gensim.models import HdpModel
from nltk.corpus import stopwords
from scipy.sparse import dok_matrix
from sklearn.feature_extraction.text import CountVectorizer
from numba import jit, prange


def prep_csv():
    '''Read the CSV from the disk and divide it into chunks.'''

    chunkdir = Path(ARGS.filename).stem + '-prep_csv'
    if not os.path.exists(chunkdir):
        os.mkdir(chunkdir)

        # Load CSV
        dataframe = pd.read_csv(ARGS.filename, index_col='url',
                                usecols=['url', 'content'])
        print('Number of pages:', dataframe.shape[0])

        n_batches = multiprocessing.cpu_count() \
            if ARGS.n_jobs == -1 else ARGS.n_jobs
        n_chunks = n_batches * ARGS.mul

        # Save chunks
        for chunk_id, df_i in enumerate(np.array_split(dataframe, n_chunks)):
            filename = f'{chunkdir}/{chunk_id}'
            df_i.to_parquet(filename, engine='pyarrow')

    return (chunkdir,)


def get_text(filename):
    '''Parse a block of pages.'''

    block = pd.read_parquet(filename, engine='pyarrow')
    content = block['content'].values

    raw = [None]*len(content)  # Preallocate to avoid resizing
    for idx, page in enumerate(content):
        soup = BeautifulSoup(page, 'html5lib')  # Parse HTML

        # Remove CSS & JS
        for script in soup(['script', 'style']):
            script.decompose()

        raw[idx] = soup.get_text()  # Get raw text
    block['content'] = raw

    parsedir = Path(ARGS.filename).stem + '-parse_html'
    stemname = Path(filename).stem
    new_filename = f'{parsedir}/{stemname}'
    block.to_parquet(new_filename, engine='pyarrow')


def parse_html(chunkdir):
    '''Keep just the text from data.'''

    parsedir = Path(ARGS.filename).stem + '-parse_html'
    if not os.path.exists(parsedir):
        os.mkdir(parsedir)

        n_batches = multiprocessing.cpu_count() \
            if ARGS.n_jobs == -1 else ARGS.n_jobs

        pool = multiprocessing.Pool(processes=n_batches, maxtasksperchild=1)

        filenames = [
            f'{chunkdir}/{tmpfile}' for tmpfile in os.listdir(chunkdir)]
        pool.map(get_text, filenames, chunksize=1)
        pool.close()
        pool.join()

    raw_files = Path(parsedir).glob('*')
    raw = pd.concat(
        pd.read_parquet(raw_slice, engine='pyarrow')
        for raw_slice in raw_files
    )['content']

    return (raw,)


def bow(raw):
    '''Convert the collection of texts in a BOW format.'''

    vectorizer = CountVectorizer(
        stop_words=stopwords.words(ARGS.lang),
        lowercase=True, strip_accents='unicode',
        dtype=np.int32)
    bow_data = vectorizer.fit_transform(raw)
    vocab = dict([v, k] for k, v in vectorizer.vocabulary_.items())

    print('Number of words:', len(vocab))

    return (bow_data, vocab)


def hda(bow_data, vocab):
    '''Perform HDA on the dataset.'''

    corpus = Sparse2Corpus(bow_data, documents_columns=False)
    hdp = HdpModel(corpus, vocab)

    # Trasform doctopic into a matrix
    doctopic = corpus2dense(hdp[corpus], num_terms=hdp.m_T,
                            num_docs=bow_data.shape[0],
                            dtype=np.float32)

    return (np.transpose(doctopic), )


def similarity_graph_dot(doctopic):
    '''Compute the similarity graph using the dot product.'''

    dot = doctopic.dot(doctopic.transpose())
    return (dot.todense(),)


def similarity_graph(doctopic):
    '''Compute the similarity graph using the Hellinger distance.'''

    n_docs = doctopic.shape[0]

    res = np.zeros((n_docs, n_docs), dtype=np.float32)
    hellinger_parallel(doctopic, res)
    return (res, )


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def hellinger_parallel(doctopic, res):
    '''Use Numba to speedup computations of similarity_graph().'''

    sqdoc = np.sqrt(doctopic)
    for row_i_idx in prange(doctopic.shape[0]):
        row_i = sqdoc[row_i_idx]

        for row_j_idx in prange(row_i_idx):
            row_j = sqdoc[row_j_idx]

            # Compute the distance
            sub_ij = row_i - row_j
            pow_ij = np.power(sub_ij, 2)
            sum_ij = pow_ij.sum()
            sq_ij = math.sqrt(sum_ij)
            res_ij = sq_ij / np.sqrt(2)

            # Store the results
            res[row_i_idx, row_j_idx] = res_ij
            res[row_j_idx, row_i_idx] = res_ij


def plot(sim_graph):
    '''Plot the similarity graph.'''

    plt.axis('off')
    plt.imshow(sim_graph, cmap='YlOrBr_r')
    plt.clim(0, 1)
    plt.colorbar().ax.invert_yaxis()
    plt.title(ARGS.filename)
    plt.savefig(ARGS.filename + '.svg')


# The execution starts here
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('filename',
                        help='filename of the CSV containing the data')
    PARSER.add_argument('lang', help='Language of the corpus')
    PARSER.add_argument('--n_jobs', type=int, default=-1,
                        help='number of jobs to run in parallel; ' +
                        'default uses all cpus')
    PARSER.add_argument('--mul', type=int, default=1,
                        help='Tradeoff between ' +
                        'performance (lower vals) and ' +
                        'memory consumption')
    ARGS = PARSER.parse_args()

    nltk.download('stopwords')

    params = ()
    STEPS = [prep_csv, parse_html, bow, hda, similarity_graph, plot]
    for step in STEPS:
        print('Running step', step.__name__)

        start = time.time()
        params = step(*params)
        end = time.time()
        print(step.__name__, 'finished in', end-start, 'sec')

        # Try to release as much memory as possible
        gc.collect()
