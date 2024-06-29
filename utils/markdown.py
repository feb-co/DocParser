import re
import copy


def markdown_text(sections: list) -> list:
    secs = copy.deepcopy(sections)
    for sec in enumerate(secs):
        if sec.get('layout_type', 'text') == 'title':
            if sec.get('layoutno', '') == 'title-0':
                sec['text'] = '# ' + sec['text']
            elif sec.get('layoutno', '') == 'title-1':
                sec['text'] = '## ' + sec['text']
            elif sec.get('layoutno', '') == 'title-2':
                sec['text'] = '### ' + sec['text']
    
    return secs


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
