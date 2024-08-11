PROMPT_FORMAT = """Only correct the following content's typo (unnecessary line breaks, splice incomplete sentences, Incorrect Markdown/HTML format, Incorrect value(O%->0%)), do not change ori content, If there is table information in the content, please convert it to the correct Markdown or HTML format:

The content:
```````
{data}
```````

The output format should be:
[BEGIN]
xxx
[END]

Note:
1. The Html format table should be Html format, the Markdown format table should be Markdown format.
2. If there is a problem with the table content in HTML format, please help me correct it.
3. When the content include the placehold token: "[IMG-x]", don't delete/correct it.
4. Please generate the correction results directly according to the above format, and do not output the intermediate thinking content."""


PROMPT_FORMAT_HIS = """Based on the following History Content, Only correct the following content's typo (unnecessary line breaks, splice incomplete sentences, Incorrect Markdown/HTML format, Incorrect value(O%->0%)), do not change ori content, If there is table information in the content, please convert it to the correct Markdown/HTML format:

The History Content:
```````
{last_data}
```````

The content that need correct:
```````
{data}
```````

The output format should be:
[BEGIN]
xxx
[END]

Note:
1. The Html format table should be Html format, the Markdown format table should be Markdown format.
2. If there is a problem with the table content in HTML format, please help me correct it.
3. When the content include the placehold token: "[IMG-x]", don't delete/correct it.
4. Please generate the correction results directly according to the above format, and do not output the intermediate thinking content."""
