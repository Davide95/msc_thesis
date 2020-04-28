'''Preprocessing on the scraped data.'''

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
from pathlib import Path
import time
import pandas as pd
import os


def remove_protocol(url):
    '''Remove http:// or https:// from a given URL.'''

    assert url.startswith('http'), f'Formatting error: URL "{url}" not valid.'
    if url.startswith('http://'):
        return url[7:]
    else:
        return url[8:]


# The execution starts here
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('input_folder',
                        help='Input folder containing all CSV files')
    PARSER.add_argument('output_folder',
                        help='Output folder where you can save the results')
    ARGS = PARSER.parse_args()

    # Create output folder if not exist
    Path(ARGS.output_folder).mkdir(parents=True, exist_ok=True)

    # Process each csv one file at a time
    input_files = Path(ARGS.input_folder).glob('*.csv')
    for input_file in input_files:
        print('Processing', input_file.stem)
        start = time.time()

        print('Removing protocols from url col...')
        data = pd.read_csv(input_file)
        data['url'] = data['url'].apply(remove_protocol)

        print('Removing duplicates from url col...')
        initial_len = len(data)
        data.drop_duplicates(subset='url', keep='first', inplace=True)
        final_len = len(data)
        print('Items removed:', initial_len-final_len)

        print('Removing protocols from connected_to col...')
        if 'connected_to' in data.columns:
            for idx, urls in data['connected_to'].iteritems():
                if not pd.isnull(urls):
                    new_urls = ','.join(map(remove_protocol, urls.split(',')))
                    data['connected_to'][idx] = new_urls
        else:
            print('Column not found.')

        print('Saving on disk...')
        data.to_csv(os.path.join(ARGS.output_folder, input_file.stem + '.csv'))

        end = time.time()
        print('Finished in', end-start, 'sec\n')
