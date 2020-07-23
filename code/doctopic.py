'''Compute the doctopic matrix.'''

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
import gc
import multiprocessing
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

from gensim.matutils import Sparse2Corpus, corpus2dense
from gensim.models import HdpModel


def prep_csv():
    '''Read the CSV from the disk and divide it into chunks.'''

    chunkdir = Path(ARGS.filename).stem + '-prep_csv'
    if not os.path.exists(chunkdir):
        os.mkdir(chunkdir)

        # Load CSV
        dataframe = pd.read_csv(ARGS.filename,
                                usecols=['url', 'content'])
        print('Number of pages:', dataframe.shape[0])

        n_batches = multiprocessing.cpu_count() \
            if ARGS.n_jobs == -1 else ARGS.n_jobs
        n_chunks = n_batches * ARGS.mul

        # Save chunks
        for chunk_id, df_i in enumerate(np.array_split(dataframe, n_chunks)):
            filename = f'{chunkdir}/{chunk_id}'
            df_i.to_parquet(filename, engine='pyarrow', index=False)

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
    block.to_parquet(new_filename, engine='pyarrow', index=False)


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
    raw_files_sorted = sorted([int(raw_slice.stem) for raw_slice in raw_files])
    raw = pd.concat(
        pd.read_parquet(f'{parsedir}/{file}', engine='pyarrow')
        for file in raw_files_sorted
    )['content']

    return (raw,)


def read_bow():
    '''Read BOW instead of computing it.'''
    dataframe = pd.read_csv(ARGS.filename, usecols=['content'])

    content = list(map(lambda x: np.fromstring(x, dtype=np.int32, sep=','),
                       dataframe['content'].values))
    content = np.asarray(content)

    # Since we don't have a vocabulary, it's a 1:1 map with the indexes
    vocab = dict([k, k] for k in range(content[0].shape[0]))

    return (sparse.csr_matrix(content), vocab)


def bow(raw):
    '''Convert the collection of texts in a BOW format.'''

    vectorizer = CountVectorizer(
        stop_words=stopwords.words(ARGS.lang),
        lowercase=True, strip_accents='unicode',
        max_df=ARGS.max_df, min_df=ARGS.min_df,
        dtype=np.int32)
    bow_data = vectorizer.fit_transform(raw)
    vocab = dict([v, k] for k, v in vectorizer.vocabulary_.items())

    print('Number of words:', len(vocab))

    return (bow_data, vocab)


def hda(bow_data, vocab):
    '''Perform HDA on the dataset.'''

    corpus = Sparse2Corpus(bow_data, documents_columns=False)
    hdp = HdpModel(corpus, vocab, max_time=ARGS.max_time)

    # Trasform doctopic into a matrix
    inference = [hdp.__getitem__(document, eps=0.0) for document in corpus]
    doctopic = corpus2dense(inference, num_terms=hdp.m_T,
                            num_docs=bow_data.shape[0],
                            dtype=np.float32)

    out = np.transpose(doctopic)
    np.save(Path(ARGS.filename).stem + '-doctopic.npy', out)

    return (out, )


def plot_topic_importance(doctopic):
    '''Plot the importance of each topic.'''

    if ARGS.plot_topics is not None:
        bins_idx = np.arange(doctopic.shape[1])
        bins_int = np.average(doctopic, axis=0)

        plt.figure(1)
        plt.bar(bins_idx, bins_int)
        plt.tick_params(axis='x',
                        which='both',
                        bottom=False,
                        top=False,
                        labelbottom=False)
        plt.ylabel('Average value')
        plt.xlabel('Topic index')
        plt.ylim([0, 1])
        plt.title(ARGS.dataset_id)
        plt.savefig(ARGS.plot_topics)

    return (doctopic, )


# The execution starts here
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('filename',
                        help='filename of the CSV containing the data')
    PARSER.add_argument('lang', help='Language of the corpus')
    PARSER.add_argument('--n_jobs', type=int, default=-1,
                        help='number of jobs to run in parallel; ' +
                        'default uses all cpus')
    PARSER.add_argument('--mul', type=int, default=1,
                        help='Trade-off between ' +
                        'performance (lower vals) and ' +
                        'memory consumption')
    PARSER.add_argument('--max_df', type=float, default=1.0,
                        help='Frequency threshold for stopwords')
    PARSER.add_argument('--min_df', type=int, default=1,
                        help='Cut-off / minimum counter threshold')
    PARSER.add_argument('--plot_topics', type=str, default=None,
                        help='Path where to plot the topics importance')
    PARSER.add_argument('--dataset_id', type=str, default='',
                        help='Title of the plots')
    PARSER.add_argument('--max_time', type=int, default=3600,
                        help='Maximum number of seconds of training time')
    PARSER.add_argument('--skip_parsing', action='store_true',
                        help='Skip the HTML parsing if the content is in the BOW format')
    ARGS = PARSER.parse_args()

    nltk.download('stopwords')

    PARAMS = ()

    if not ARGS.skip_parsing:
        STEPS = [prep_csv, parse_html, bow, hda, plot_topic_importance]
    else:
        STEPS = [read_bow, hda, plot_topic_importance]

    for step in STEPS:
        print('Running step', step.__name__)

        start = time.time()
        PARAMS = step(*PARAMS)
        end = time.time()
        print(step.__name__, 'finished in', end-start, 'sec')

        # Try to release as much memory as possible
        gc.collect()
