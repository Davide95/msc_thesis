'''Compute the similarity graph.'''

import argparse
import gc
import logging
import multiprocessing
import os
import time
from pathlib import Path

import dask.dataframe as dd
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from gensim.matutils import Sparse2Corpus
from gensim.models import HdpModel
from nltk.corpus import stopwords
from scipy.sparse import dok_matrix
from sklearn.feature_extraction.text import CountVectorizer


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

    raw = dd.read_parquet(f'{parsedir}/*',
                          engine='pyarrow')['content'].compute()

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

    # Trasform doctopic into a sparse matrix
    doctopic_sparse = dok_matrix(
        (bow_data.shape[0], hdp.m_T), dtype=np.float32)

    for row in range(bow_data.shape[0]):
        doctopic = hdp[corpus[row]]

        if doctopic == []:
            logging.warning('Document in row %s has no topics', row)
        else:
            cols, elems = zip(*doctopic)
            cols, elems = list(cols), list(elems)
            for pos_col, col in enumerate(cols):
                doctopic_sparse[row, col] = elems[pos_col]

    doctopic_sparse = doctopic_sparse.tocsc()
    return (doctopic_sparse,)


def similarity_graph(doctopic):
    '''Compute the similarity graph of the document-topic matrix.'''

    dot = doctopic.dot(doctopic.transpose())
    return (dot,)


def plot(sim_graph):
    '''Plot the similarity graph.'''
    plt.imshow(sim_graph.todense())
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
