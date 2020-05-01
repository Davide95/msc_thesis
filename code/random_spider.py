'''Web Crawling of random content.'''

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

import time
import scrapy


class RandomSpider(scrapy.Spider):
    name = 'Random'
    num_pages = 1000
    start_url = 'https://en.wikipedia.org/wiki/Special:Random'
    wait_sec = 1.0  # Wait this number of seconds between each page

    def start_requests(self):
        for _ in range(self.num_pages):
            yield scrapy.Request(url=self.start_url,
                                 callback=self.parse, dont_filter=True)
            time.sleep(self.wait_sec)  # Avoid to be banned from the website

    def parse(self, response):
        yield {
            'url': response.url.replace(',', '%2C'),
            'content': response.text
        }
