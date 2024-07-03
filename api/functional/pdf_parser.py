import pdfplumber
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()


from src.parser import PdfParser


class Pdf(PdfParser):
    def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, zoomin=3, callback=None):
        callback(0.0, msg="OCR is running...")
        self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            callback)
        callback(0.5 , msg="OCR finished")

        self._layouts_rec(zoomin)

        callback(0.67, "Layout analysis finished")

        self._table_transformer_job(zoomin)

        callback(0.68, "Table analysis finished")

        self._text_merge()
        tbls = self._extract_table_figure(True, zoomin, True, True)
        self._naive_vertical_merge()
        self._filter_forpages()
        self._merge_with_same_bullet()

        callback(0.75, "Text merging finished.")
        callback(0.8, "Text extraction finished")

        return self.boxes, tbls


def parser(file_path, from_page, to_page, callback=None, postprocess=None):
    pdf_parser = Pdf()
    sections, tables = pdf_parser(
        file_path,
        from_page=from_page,
        to_page=to_page,
        callback=callback
    )
    return postprocess(sections, tables)


def get_document_total_pages(file_path, tokenizer_fn, chunk_token_size) -> int:
    pdf_page_number = Pdf.total_page_number(
        file_path,
        binary=not isinstance(file_path, str)
    )
    return pdf_page_number


if __name__ == "__main__":
    import sys
    
    def dummy(prog=None, msg=""):
        # print('>>>>>', prog, msg, flush=True)
        pass
    
    def postprocess(sections, tables):
        for sec in sections:
            print(sec)

    res = parser(
        sys.argv[1],
        from_page=0,
        to_page=1,
        callback=dummy
    )
