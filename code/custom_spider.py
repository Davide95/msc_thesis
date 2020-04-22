'''Web Crawling of the desired website.'''

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

import scrapy
from scrapy.linkextractors import LinkExtractor


class CustomSpider(scrapy.Spider):
    name = 'Custom'
    start_urls = ['https://www.example.org/']
    allowed_domains = ['www.example.org']

    def parse(self, response):
        links = list(LinkExtractor(
            allow_domains=self.allowed_domains).extract_links(response))

        yield {
            'url': response.url.replace(',', '%2C'),
            'connected_to': [response.urljoin(url.url.replace(',', '%2C'))
                             for url in links],
            'content': response.text
        }

        for link in links:
            yield response.follow(link, callback=self.parse)
