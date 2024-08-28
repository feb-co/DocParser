import re

def is_url(input_str):
    # 使用正则表达式检查输入是否为URL
    url_regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https:// or ftp://
        r'(?:\S+(?::\S*)?@)?'  # 用户和密码
        r'(?:[A-Za-z0-9.-]+|\[[0-9a-fA-F:]+\])'  # 域名
        r'(?::\d+)?'  # 端口
        r'(?:/\S*)?$'  # 路径
    )
    return re.match(url_regex, input_str) is not None
