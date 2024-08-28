import os
import copy
import json
import logging
from enum import Enum
from dataclasses import dataclass, field

from docparser_feb.scripts.log_level import LOGING_MAP
from docparser_feb.scripts.openai import openai


log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.getLogger().setLevel(LOGING_MAP[log_level])


@dataclass
class HtmlObject:
    mode = None
    page_texts: list = field(default_factory=list)
    images: dict = field(default_factory=dict)
    data_json: dict = field(default_factory=dict)
    page_images: list = field(default_factory=list)


class HtmlPostprocess(object):
    def __init__(self, use_llm: bool = False) -> None:
        self.use_llm = use_llm

        self.__page_texts = None
        self.__page_images = None

    def postprocess(self, page_texts, url_list=None):
        self.__page_texts = []
        for i in range(len(page_texts)):
            self.__page_texts.append(
                {
                    "title": page_texts[i].split("\n")[0],
                    "content": page_texts[i]
                    .replace("\n\n\n", "\n\n")
                    .replace("\n\n\n", "\n\n")
                    .replace("\n\n\n", "\n\n"),
                }
            )

        logging.info("pdf rule-based postprocess finished ...")
        if self.use_llm:
            for i in range(len(page_texts)):
                self.__page_texts[i]["content"] = openai.format_data(
                    self.__page_texts[i]["content"]
                )
            logging.info("pdf LLM formatting finished ...")

        return HtmlObject(
            page_texts=self.__page_texts,
        )

    def save_data(self, html_object: HtmlObject, output_dir: str):
        if output_dir:
            for i in range(len(html_object.page_texts)):
                title = html_object.page_texts[i]["title"].replace("/", "-")
                open(f"{output_dir}/{title}.md", "w", encoding="utf-8").write(
                    html_object.page_texts[i]["content"]
                )
        else:
            for i in range(len(html_object.page_texts)):
                title = html_object.page_texts[i]["title"]
                content = html_object.page_texts[i]["content"]
                print(f"Title: {title}")
                print(f"Content: {content}")
