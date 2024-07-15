import re
import copy
import pandas as pd
from bs4 import BeautifulSoup


def can_convert_to_markdown(html_table):
    soup = BeautifulSoup(html_table, "html.parser")
    table = soup.find("table")

    # 检测是否存在合并的单元格
    if table.find_all(["th", "td"], attrs={"rowspan": True}) or table.find_all(
        ["th", "td"], attrs={"colspan": True}
    ):
        return False

    # 检测是否包含内嵌的HTML元素
    if any(tag.name in ["div", "span", "img", "a"] for tag in table.find_all()):
        return False

    return True


def html_to_markdown(html_table):
    # 检查是否可以转换
    can_convert = can_convert_to_markdown(html_table)
    if not can_convert:
        return html_table

    # 使用Pandas转换表格
    df = pd.read_html(html_table)[0]
    return df.to_markdown(index=False)


def markdown_text(boxes: list) -> list:
    new_boxes = copy.deepcopy(boxes)
    for box in new_boxes:
        if box.get("layout_type", "text") == "text":
            box["text"] = box["text"].strip().replace("   ", " ").replace("  ", " ")

        if box.get("layout_type", "text") == "title":
            box["text"] = box["text"].strip().replace("   ", " ").replace("  ", " ")
            if box.get("layoutno", "") == "title-0":
                box["text"] = "# " + box["text"]
            elif box.get("layoutno", "") == "title-1":
                box["text"] = "## " + box["text"]
            elif box.get("layoutno", "") == "title-2":
                box["text"] = "### " + box["text"]

        if box.get("layout_type", "text") == "reference":
            box["text"] = box["text"].strip().replace("   ", " ").replace("  ", " ")
            if box.get("layoutno", "") == "reference-0":
                box["text"] = "- " + box["text"]
            elif box.get("layoutno", "") == "reference-1":
                box["text"] = "---\n\n" + box["text"]

    return new_boxes


def markdown_table(boxes: list) -> list:
    new_boxes = copy.deepcopy(boxes)
    for box in new_boxes:
        if box.get("layout_type", "text") == "table":
            box["text"] = html_to_markdown(box["text"])

    return new_boxes


def markdown_equation(s: str) -> str:
    # equation tag
    s = re.sub(
        r"^\(([\d.]+[a-zA-Z]?)\) \\\[(.+?)\\\]$", r"\[\2 \\tag{\1}\]", s, flags=re.M
    )
    s = re.sub(
        r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\)$", r"\[\1 \\tag{\2}\]", s, flags=re.M
    )
    s = re.sub(
        r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\) (\\\[.+?\\\])$",
        r"\[\1 \\tag{\2}\] \3",
        s,
        flags=re.M,
    )  # multi line
    s = s.replace(r"\. ", ". ")
    # bold formatting
    s = s.replace(r"\bm{", r"\mathbf{").replace(r"{\\bm ", r"\mathbf{")
    # s = s.replace(r"\it{", r"\mathit{").replace(r"{\\it ", r"\mathit{") # not needed
    s = re.sub(r"\\mbox{ ?\\boldmath\$(.*?)\$}", r"\\mathbf{\1}", s)
    # s=re.sub(r'\\begin{table}(.+?)\\end{table}\nTable \d+: (.+?)\n',r'\\begin{table}\1\n\\capation{\2}\n\\end{table}\n',s,flags=re.S)
    # s=re.sub(r'###### Abstract\n(.*?)\n\n',r'\\begin{abstract}\n\1\n\\end{abstract}\n\n',s,flags=re.S)
    # urls
    s = re.sub(
        r"((?:http|ftp|https):\/\/(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-]))",
        r"[\1](\1)",
        s,
    )
    # algorithms
    s = re.sub(r"```\s*(.+?)\s*```", r"```\n\1\n```", s, flags=re.S)
    # lists

    return s


def format_table(boxes: list) -> list:
    new_boxes = copy.deepcopy(boxes)
    for box in new_boxes:
        if box.get("layout_type", "text") == "table":
            box["text"] = (
                box["text"]
                .replace("<td   >", "<td>")
                .replace("<th   >", "<th>")
                .replace("<td  >", "<td>")
                .replace("<th  >", "<th>")
            )

    return new_boxes
