'''Web Crawling of the desired website.'''

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
            'url': response.url,
            'connected_to': [response.urljoin(url.url) for url in links],
            'content': response.text
        }

        for link in links:
            yield response.follow(link, callback=self.parse)
