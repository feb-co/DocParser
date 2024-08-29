import os
import asyncio

from docparser_feb.src.parser.html_parser import HtmlParser


os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


class FebHtmlParser(HtmlParser):
    def __call__(self, url_list: list, async_mode: bool = False):
        if async_mode:
            return self._async_call(url_list)
        else:
            return self._sync_call(url_list)
    
    def _sync_call(self, url_list: list):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环已经在运行，使用新的事件循环
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            results = new_loop.run_until_complete(self._async_call(url_list))
            new_loop.close()
            return results
        else:
            return loop.run_until_complete(self._async_call(url_list))
    
    async def _async_call(self, url_list: list):
        results = await self.ascrape_all_page(url_list)
        return results


def main():
    import argparse
    from docparser_feb.scripts.string_utils import is_url
    from docparser_feb.scripts.file_utils import is_file, read_urls_from_file
    from docparser_feb.scripts.postprocess import HtmlPostprocess

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
    parser.add_argument(
        "--fast",
        action="store_true",
        help="get text without markdown.",
    )
    args = parser.parse_args()

    if is_url(args.input):
        url_list = [args.input]
    elif is_file(args.input):
        url_list = read_urls_from_file(args.input)
    else:
        raise TypeError("Not support other file types, only support url or a .txt file")

    html_parser = FebHtmlParser(
        os.getenv("JINA_KEY", ""),
        is_cache=True,
        timeout=10,
        disable_tqdm=False,
        fast=args.fast
    )
    page_texts = asyncio.run(html_parser(url_list, True))
    post_process_obj = HtmlPostprocess(
        use_llm=bool(args.use_llm),
    )
    html_object = post_process_obj.postprocess(page_texts)
    post_process_obj.save_data(html_object, args.output_dir)


if __name__ == "__main__":
    main()
