import os
import asyncio
import aiohttp
import logging
from tqdm import tqdm
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

from docparser_feb.scripts.log_level import LOGING_MAP

log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.getLogger().setLevel(LOGING_MAP[log_level])


class HtmlParser:
    def __init__(self, jina_key: str, is_cache=True, timeout=10, disable_tqdm=False, fast=False):
        self.fast = fast
        self.jina_prefix = "https://r.jina.ai/" if not fast else ""
        self.jina_key = jina_key

        self.headers = {
            "X-Return-Format": "markdown",
        }
        if self.jina_key:
            self.headers["Authorization"] = f"Bearer {self.jina_key}"
        if not is_cache:
            self.headers["X-No-Cache"] = "true"

        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.disable_tqdm = disable_tqdm

    def extract_core_text(self, html_content):
        if html_content is not None:
            soup = BeautifulSoup(html_content, 'html.parser')
            core_text = ""
            for p in soup.find_all('p'):
                core_text += p.get_text() + "\n"
            return core_text.strip()
        else:
            return None

    def scrape_page(self, url: str):
        response_text = asyncio.run(self.ascrape_page(url))
        return response_text

    async def ascrape_page(self, url: str, try_time=0):
        if try_time > 3:
            return None

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(
                    self.jina_prefix + url, headers=self.headers
                ) as response:
                    response_text = await response.text()
                    if self.fast:
                        response_text = self.extract_core_text(response_text)
                    return response_text
        except Exception as e:
            return await self.ascrape_page(url, try_time + 1)

    def scrape_all_page(self, urls: list):
        results = []
        if self.disable_tqdm:
            tqdm_urls = urls
        else:
            tqdm_urls = tqdm(urls, desc="scrape pages", ncols=100)
        for url in tqdm_urls:
            result = self.scrape_page(url)
            results.append(result)
        return results

    async def ascrape_all_page(self, urls: list):
        tasks = [self.ascrape_page(url) for url in urls]
        if self.disable_tqdm:
            results = await asyncio.gather(*tasks)
        else:
            results = await tqdm_asyncio.gather(*tasks, desc="scrape pages", ncols=100)

        return results
