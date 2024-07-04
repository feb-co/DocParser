import os
import sys

work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_dir)

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
        tables_figures = self._extract_table_figure(True, zoomin, True)
        self._naive_vertical_merge()
        self._filter_forpages()
        self._merge_with_same_bullet()

        callback(0.75, "Text merging finished.")
        callback(0.8, "Text extraction finished")

        return self.boxes, tables_figures


def parser(file_path, from_page, to_page, callback=None, postprocess=None):
    pdf_parser = Pdf()
    sections, tables_figures = pdf_parser(
        file_path,
        from_page=from_page,
        to_page=to_page,
        callback=callback
    )
    results = sorted(
        [bxs for bxs in sections+tables_figures],
        key=lambda x: (x["top"], x["x0"]),
    )
    return postprocess(results)


def get_document_total_pages(file_path, tokenizer_fn, chunk_token_size) -> int:
    pdf_page_number = Pdf.total_page_number(
        file_path,
        binary=not isinstance(file_path, str)
    )
    return pdf_page_number


if __name__ == "__main__":
    import sys
    from scripts import markdown
    
    def dummy(prog=None, msg=""):
        # print('>>>>>', prog, msg, flush=True)
        pass

    def postprocess(results):
        results = markdown.markdown_text(results)
        # results = markdown.markdown_table(results)
        print('\n\n'.join([res['text'] for res in results]))        

    res = parser(
        sys.argv[1],
        from_page=1,
        to_page=2,
        callback=dummy,
        postprocess=postprocess
    )
