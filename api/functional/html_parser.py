import os
import sys

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

work_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(work_dir)

from src.parser import HtmlParser


def parser_html(
    url_list: list,
    is_cache=True,
    timeout=10,
    disable_tqdm=False,
):
    parser = HtmlParser(
        os.getenv("JINA_KEY", ""),
        is_cache,
        timeout,
        disable_tqdm,
    )
    results = parser.scrape_all_page(url_list)
    return results


async def aparser_html(
    url_list: list,
    is_cache=True,
    timeout=10,
    disable_tqdm=False,
):
    parser = HtmlParser(
        os.getenv("JINA_KEY", ""),
        is_cache,
        timeout,
        disable_tqdm,
    )
    results = await parser.ascrape_all_page(url_list)
    return results


def main():
    import argparse
    from scripts.string_utils import is_url
    from scripts.file_utils import is_file, read_urls_from_file
    from scripts.postprocess import HtmlPostprocess

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="TXT files used to store HTML web pages, or single web page link.",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory where to store the output results (md/json/images).",
        default="",
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

    if is_url(args.input):
        url_list = [args.input]
    elif is_file(args.input):
        url_list = read_urls_from_file(args.input)
    else:
        raise TypeError("Not support other file types, only support url or a .txt file")

    page_texts = parser_html(url_list)
    post_process_obj = HtmlPostprocess(
        use_llm=bool(args.use_llm),
    )
    html_object = post_process_obj.postprocess(page_texts)
    post_process_obj.save_data(html_object, args.output_dir)


if __name__ == "__main__":
    main()
