'''Compute the hellinger distance of a doctopic matrix.'''

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
import math
from pathlib import Path

import numpy as np
from numba import jit, prange


def hellinger_distance(doctopic):
    '''Computing distances using the Hellinger distance.'''

    n_docs = doctopic.shape[0]

    res = np.zeros((n_docs, n_docs), dtype=np.float32)
    hellinger_parallel(doctopic, res)

    return res


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def hellinger_parallel(doctopic, res):
    '''Hellinger distance over multiple cores.'''

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


def js_distance(doctopic):
    '''Computing distances using the Jensen-Shannon divergence.'''

    n_docs = doctopic.shape[0]

    res = np.zeros((n_docs, n_docs), dtype=np.float32)
    js_parallel(doctopic, res)

    return res


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def js_parallel(doctopic, res):
    '''Jensen-Shannon divergence over multiple cores.'''

    for row_i_idx in prange(doctopic.shape[0]):
        row_i = doctopic[row_i_idx]

        for row_j_idx in prange(row_i_idx):
            row_j = doctopic[row_j_idx]
            m = (row_i + row_j) / 2.0

            # Compute the divergence
            kl_ij = kl_divergence(row_i, m)
            kl_ji = kl_divergence(row_j, m)
            divergence = (kl_ij + kl_ji) / 2.0

            # Store the results
            res[row_i_idx, row_j_idx] = divergence
            res[row_j_idx, row_i_idx] = divergence


@jit(nopython=True, nogil=True, fastmath=True)
def kl_divergence(p, q):
    '''Hellinger distance over multiple cores.'''

    log_2 = np.log(p / q) / np.log(2)
    res_vect = np.multiply(p, log_2)
    return np.sum(res_vect)


# The execution starts here
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('filename',
                        help='filename of the doctopic matrix in .npy format')
    ARGS = PARSER.parse_args()
    FILENAME_ONLY = Path(ARGS.filename).stem

    DOCTOPIC = np.load(ARGS.filename)

    print('Computing the Hellinger distance matrix...')
    HD_MATRIX = hellinger_distance(DOCTOPIC)
    np.save(FILENAME_ONLY + '-hd.npy', HD_MATRIX)

    print('Computing the Jensen-Shannon distance matrix...')
    JS_MATRIX = js_distance(DOCTOPIC)
    np.save(FILENAME_ONLY + '-js.npy', JS_MATRIX)

    print('Finished.')
