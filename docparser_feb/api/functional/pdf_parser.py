import os
import sys

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
from docparser_feb.src.parser.pdf_parser import PdfParser


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

        results = self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            is_english=is_english,
        )
        if not results:
            return None
            
        callback(0.3, msg="OCR detect finished")

        self._layouts_rec()
        callback(0.4, "Layout analysis finished")

        self._text_merge()
        self._naive_vertical_merge()
        callback(0.5, "Text merging finished.")

        self._filter_forpages()
        self._merge_with_same_bullet()
        callback(0.6, "Text extraction finished")

        self._text_predict()
        callback(0.7, msg="Nougat predict finished")

        tables = self._extract_table_figure(True)
        callback(0.9, msg="table figure process finished")

        results = sorted(
            [bxs for bxs in self.boxes + tables],
            key=lambda x: (x["page_number"], x["top"], x["x0"]),
        )

        return results


def parser_pdf(
    file_path, from_page, to_page, callback=None, postprocess=None, is_english=None
):
    pdf_parser = Pdf()
    results = pdf_parser(
        file_path,
        from_page=from_page,
        to_page=to_page,
        callback=callback,
        is_english=is_english,
    )
    if results:
        return postprocess(results, pdf_parser.page_images)
    else:
        return None


def get_document_total_pages(
    file_path, tokenizer_fn=None, chunk_token_size=None
) -> int:
    pdf_page_number = Pdf.total_page_number(
        file_path, binary=not isinstance(file_path, str)
    )
    return pdf_page_number


def main():
    import argparse
    from docparser_feb.scripts.file_utils import init_file_path
    from docparser_feb.scripts.postprocess import PdfPostprocess, PdfMode

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        help="Directory where to store PDFs, or a file path to a single PDF",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Directory where to store the output results (md/json/images).",
        required=True,
    )
    parser.add_argument(
        "--page_range",
        help="The page range to parse the PDF, the format is 'start_page:end_page', that is, [start, end). Default: full.",
        default="full",
    )
    parser.add_argument(
        "--mode",
        help="The mode for parsing the PDF, to extract only the plain text or the text plus images.",
        choices=["plain", "figure placehold", "figure latex"],
        default="plain",
    )
    parser.add_argument(
        "--rendering",
        action="store_true",
        help="Is it necessary to render the recognition results of the input PDF to output the recognition range? Default: False.",
    )
    parser.add_argument(
        "--use_llm",
        action="store_true",
        help="Do you need to use LLM to format the parsing results? If so, please specify the corresponding parameters through the environment variables: DOC_PARSER_OPENAI_URL, DOC_PARSER_OPENAI_KEY, DOC_PARSER_OPENAI_MODEL. Default: False.",
    )
    parser.add_argument(
        "--overwrite_result",
        action="store_true",
        help="If the parsed target file already exists, should it be rewritten? Default: False.",
    )
    args = parser.parse_args()

    def dummy(prog=None, msg=""):
        print(">>>>>>>>", prog, msg, flush=True)
        pass

    files = init_file_path(args.inputs, ".pdf")
    if args.mode == "plain":
        mode = PdfMode.PlainText
    elif args.mode == "figure placehold":
        mode = PdfMode.FigurePlacehold
    elif args.mode == "figure latex":
        mode = PdfMode.FigureLatex
    else:
        raise NotImplementedError

    for file in files:
        post_process_obj = PdfPostprocess(
            mode=mode,
            rendering=bool(args.rendering),
            use_llm=bool(args.use_llm),
        )
        if ":" in args.page_range:
            start_page, end_page = args.page_range.split(":")
        else:
            start_page = 0
            end_page = get_document_total_pages(file)

        if not args.overwrite_result and post_process_obj.file_exist(
            file, args.output_dir
        ):
            print(f"The parser result ({file}) exist, will skip...", flush=True)
            continue

        pdf_object = parser_pdf(
            file,
            from_page=int(start_page),
            to_page=int(end_page),
            callback=dummy,
            postprocess=post_process_obj.postprocess,
        )
        if pdf_object:
            post_process_obj.save_data(pdf_object, file, args.output_dir)
        else:
            continue


if __name__ == "__main__":
    main()
