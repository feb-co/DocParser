import os
import sys

work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_dir)

from dotenv import load_dotenv

load_dotenv()


from src.parser import PdfParser


class Pdf(PdfParser):
    def __call__(
        self,
        filename,
        binary=None,
        from_page=0,
        to_page=100000,
        zoomin=3,
        callback=None,
        is_english=None,
    ):
        callback(0.0, msg="Parser is running...")

        self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            callback,
            is_english=is_english,
        )
        callback(0.3, msg="OCR detect finished")

        self._layouts_rec()
        callback(0.4, "Layout analysis finished")

        self._text_merge()
        self._naive_vertical_merge()
        callback(0.5, "Text merging finished.")

        self._filter_forpages()
        self._merge_with_same_bullet()
        callback(0.6, "Text extraction finished")

        self._text_predict(zoomin)
        callback(0.7, msg="Nougat predict finished")

        tables = self._extract_table_figure(False, True)
        callback(0.9, msg="table figure process finished")

        results = sorted(
            [bxs for bxs in self.boxes + tables],
            key=lambda x: (x["top"], x["x0"]),
        )
        return results


def parser(
    file_path, from_page, to_page, callback=None, postprocess=None, is_english=None
):
    pdf_parser = Pdf()
    results = []
    for page_i in range(from_page, to_page):
        results += pdf_parser(
            file_path,
            from_page=page_i,
            to_page=page_i + 1,
            callback=callback,
            is_english=is_english,
        )
    return postprocess(results, pdf_parser.page_images)


def get_document_total_pages(
    file_path, tokenizer_fn=None, chunk_token_size=None
) -> int:
    pdf_page_number = Pdf.total_page_number(
        file_path, binary=not isinstance(file_path, str)
    )
    return pdf_page_number


if __name__ == "__main__":
    import sys
    import json
    from scripts import markdown
    from scripts.rendering import pdf_rendering

    def dummy(prog=None, msg=""):
        # print('>>>>>', prog, msg, flush=True)
        pass

    def postprocess(results, page_images):
        results = markdown.markdown_text(results)
        # results = markdown.markdown_table(results)
        page_images = pdf_rendering.pdf_rendering(
            page_images,
            results,
        )
        page_images[0].save(
            "/home/licheng/output.pdf", save_all=True, append_images=page_images[1:]
        )
        json.dump(
            [item for item in results if item["layout_type"] != "figure"],
            open("/home/licheng/output.json", "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )

    res = parser(
        sys.argv[1], from_page=1, to_page=2, callback=dummy, postprocess=postprocess
    )
