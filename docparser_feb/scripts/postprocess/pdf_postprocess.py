import os
import copy
import json
import logging
from enum import Enum
from dataclasses import dataclass, field

from docparser_feb.scripts.log_level import LOGING_MAP
from docparser_feb.scripts import markdown
from docparser_feb.scripts.rendering import pdf_rendering
from docparser_feb.scripts.openai import openai

log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.getLogger().setLevel(LOGING_MAP[log_level])


class PdfMode(Enum):
    PlainText = "plain text"
    FigurePlacehold = "figure placehold"
    FigureLatex = "figure latex"


@dataclass
class PdfObject:
    mode: PdfMode = None
    texts: str = None
    images: dict = field(default_factory=dict)
    data_json: dict = field(default_factory=dict)
    page_images: list = field(default_factory=list)


class PdfPostprocess(object):
    def __init__(
        self,
        mode: PdfMode,
        rendering: bool = False,
        use_llm: bool = False,
    ) -> None:
        self.mode = mode
        self.rendering = rendering
        self.use_llm = use_llm

        self.__texts = None
        self.__data_json = None
        self.__images = {}
        self.__page_images = None

    def postprocess(self, results, page_images):
        new_results = []
        img_index = 0
        for item in results:
            if "figure" not in item["layout_type"]:
                new_results.append(copy.deepcopy(item))
            elif "figure" in item["layout_type"]:
                new_item = copy.deepcopy(item)
                new_item["text"] = f"[IMG-{img_index}]"
                new_results.append(new_item)
                img_index += 1
                if self.mode in (PdfMode.FigurePlacehold, PdfMode.FigureLatex):
                    image = new_item["image"]
                    self.__images[new_item["text"]] = image

        new_results = markdown.markdown_text(new_results)
        self.__texts = "\n\n".join(
            [
                str(item["text"])
                for item in new_results
                if "figure" not in item["layout_type"]
                or self.mode in (PdfMode.FigurePlacehold, PdfMode.FigureLatex)
            ]
        )

        for res in new_results:
            if "image" in res:
                del res["image"]
        self.__data_json = [
            item
            for item in new_results
            if "figure" not in item["layout_type"]
            or self.mode in (PdfMode.FigurePlacehold, PdfMode.FigureLatex)
        ]

        logging.info("pdf rule-based postprocess finished ...")
        if self.use_llm:
            self.__texts = openai.format_data(self.__texts)
            logging.info("pdf LLM formatting finished ...")

        if self.rendering:
            page_images = pdf_rendering.pdf_rendering(
                page_images,
                self.__data_json,
            )
            logging.info("pdf rendering finished ...")
        self.__page_images = page_images

        return PdfObject(
            mode=self.mode,
            texts=self.__texts,
            data_json=self.__data_json,
            images=self.__images,
            page_images=self.__page_images,
        )

    def save_data(self, pdf_object: PdfObject, input_file: str, output_dir: str):
        base_name = os.path.basename(input_file)
        file_title, _ = os.path.splitext(base_name)
        if self.rendering or self.mode in (
            PdfMode.FigurePlacehold,
            PdfMode.FigureLatex,
        ):
            full_dir_path = os.path.join(output_dir, file_title)
            try:
                os.makedirs(full_dir_path)
            except:
                pass

            if self.mode in (PdfMode.FigurePlacehold, PdfMode.FigureLatex):
                img_dir_path = os.path.join(full_dir_path, "images")
                try:
                    os.makedirs(img_dir_path)
                except:
                    pass
                for image_id in pdf_object.images:
                    image_file = f"{img_dir_path}/{image_id}.png"
                    pdf_object.images[image_id].save(image_file)
                    if self.mode in (PdfMode.FigureLatex,):
                        pdf_object.texts = pdf_object.texts.replace(
                            image_id, f"!{image_id}({image_file})"
                        )

            open(f"{full_dir_path}/{file_title}.md", "w", encoding="utf-8").write(
                pdf_object.texts
            )

            if self.rendering:
                with open(
                    f"{full_dir_path}/{file_title}.json", "w", encoding="utf-8"
                ) as fo:
                    json.dump(pdf_object.data_json, fo, ensure_ascii=False, indent=2)
                pdf_object.page_images[0].save(
                    f"{full_dir_path}/{file_title}_rendering.pdf",
                    save_all=True,
                    append_images=pdf_object.page_images[1:],
                )
        else:
            open(f"{output_dir}/{file_title}.md", "w", encoding="utf-8").write(
                pdf_object.texts
            )

    def file_exist(self, input_file: str, output_dir: str):
        base_name = os.path.basename(input_file)
        file_title, _ = os.path.splitext(base_name)
        if self.rendering or self.mode in (
            PdfMode.FigurePlacehold,
            PdfMode.FigureLatex,
        ):
            full_dir_path = os.path.join(output_dir, file_title)
        else:
            full_dir_path = f"{output_dir}/{file_title}.md"
            
        return os.path.exists(full_dir_path)
