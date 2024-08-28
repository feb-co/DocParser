import re


def split_text_by_words_num(text: str, chunk_num=2000):
    # 按照单词计数拆分文本，确保每个部分的单词数量大约为2000，同时确保句子的完整性
    split_texts = []
    words = re.split(r'(\s+)', text)  # 保留空格，以便重建文本
    word_count = 0
    current_text = ""

    for idx, word in enumerate(words):
        current_text += word
        if word.strip():  # 非空白字符计数
            word_count += 1

        # 每chunk_num个单词或文本结束时拆分
        if word_count >= chunk_num or idx == len(words)-1:
            # 确保句子完整性，查找最后一个句号并拆分
            last_period = current_text.rfind('. ')
            if last_period != -1 and last_period + 1 != len(current_text) and idx != len(words)-1:
                split_texts.append(current_text[:last_period + 1])
                current_text = current_text[last_period + 1:]
                word_count = len(current_text.split())

    split_texts.append(current_text)  # 添加最后一部分
    return split_texts
