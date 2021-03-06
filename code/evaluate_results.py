'''Evaluate results comparing the network with the structure of the website.'''

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


import logging
import argparse
import pandas as pd
from scipy.sparse import load_npz


def evaluate_data(content, slugs, urls):
    '''Evaluate all dataset.'''

    slugs = ['/' + slug for slug in slugs]

    n_tot = 0
    n_connected = 0
    for slug in slugs:
        print('Slug:', slug)

        n_connected_group, n_tot_group = evaluate_slug(urls, content, slug)
        n_tot += n_tot_group
        n_connected += n_connected_group

    print('Number of correctly identified edges:',
          n_connected / n_tot * 100, '%')


def evaluate_slug(urls, content, slug):
    '''Evaluate a single slug.'''

    n_tot_group = 0
    n_connected_group = 0
    idx_group = []
    for idx, url in enumerate(urls):
        if slug in url:

            # Check if they are connected
            for idx_from in idx_group:
                n_tot_group += 1
                if content[idx_from, idx]:
                    n_connected_group += 1
                else:
                    print(url, 'and', urls[idx_from],
                          'not connected but they should')

            idx_group.append(idx)

    print('Number of documents:', len(idx_group))
    if n_tot_group != 0:
        print('Number of correctly identified edges:',
              n_connected_group / n_tot_group * 100, '%')
    print()
    return n_connected_group, n_tot_group


# The execution starts here
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('content_filename',
                        help='filename of the final binarized matrix')
    PARSER.add_argument('slugs_filename',
                        help='filename of the json provided by the University of Genoa')
    PARSER.add_argument('preprocessed_filename',
                        help='filename of the CSV containing the already preprocessed scraped data')
    ARGS = PARSER.parse_args()

    # Load data
    CONTENT = load_npz(ARGS.content_filename).toarray()
    SLUGS = pd.read_json(ARGS.slugs_filename)['slug_it'].dropna().values
    URLS = pd.read_csv(ARGS.preprocessed_filename)['url'].values

    evaluate_data(CONTENT, SLUGS, URLS)