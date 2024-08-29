# DocParser 📄

DocParser is a powerful tool for LLM traning and other application, for examples: RAG, which support to parse multi type file, includes:

## Feature 🎉

### File types supported for parsing:

- [Pdf](#Pdf): Use OCR to parse PDF documents and output text in markdown format. The parsing results can be used for LLM pretrain, RAG, etc.
- [Html](#Html): Use [jina](https://jina.ai/reader) to parse multi html pages and output text in markdown.

## Install

From pip:

```bash
pip install docparser_feb
```

From repository:

```bash
pip install git+https://github.com/feb-co/DocParser.git
```

Or install it directly through the installation package:

```bash
git clone https://github.com/feb-co/DocParser.git
cd DocParser
pip install -e .
```

## API/Functional

### Pdf

#### From CLI

You can run the following script to get the pdf parsing results:

```shell
export LOG_LEVEL="ERROR"
export DOC_PARSER_MODEL_DIR="xxx"
export DOC_PARSER_OPENAI_URL="xxx"
export DOC_PARSER_OPENAI_KEY="xxx"
export DOC_PARSER_OPENAI_MODEL="gpt-4-0125-preview"
export DOC_PARSER_OPENAI_RETRY="3"
docparser-pdf \
    --inputs path/to/file.pdf or path/to/directory \
    --output_dir output_directory \
    --page_range '0:1' --mode 'figure latex' \
    --rendering --use_llm --overwrite_result
```

The following is a description of the relevant parameters:

```bash
usage: docparser-pdf [-h] --inputs INPUTS --output_dir OUTPUT_DIR [--page_range PAGE_RANGE] [--mode {plain,figure placehold,figure latex}] [--rendering] [--use_llm]

options:
  -h, --help            show this help message and exit
  --inputs INPUTS       Directory where to store PDFs, or a file path to a single PDF
  --output_dir OUTPUT_DIR
                        Directory where to store the output results (md/json/images).
  --page_range PAGE_RANGE
                        The page range to parse the PDF, the format is 'start_page:end_page', that is, [start, end). Default: full.
  --mode {plain,figure placehold,figure latex}
                        The mode for parsing the PDF, to extract only the plain text or the text plus images.
  --rendering           Is it necessary to render the recognition results of the input PDF to output the recognition range? Default: False.
  --use_llm             Do you need to use LLM to format the parsing results? If so, please specify the corresponding parameters through the environment variables: DOC_PARSER_OPENAI_URL, DOC_PARSER_OPENAI_KEY, DOC_PARSER_OPENAI_MODEL. Default: False.
  --overwrite_result    If the parsed target file already exists, should it be rewritten? Default: False.
```

#### From Python


### Html

#### From CLI

You can run the following script to get the html parsing results:

```bash
docparser-html https://github.com/mem0ai/mem0
```

The following is a description of the relevant parameters:

#### From Python