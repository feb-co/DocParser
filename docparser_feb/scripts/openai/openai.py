import os
import time
import requests
import logging
from tqdm import tqdm

from docparser_feb.scripts.log_level import LOGING_MAP
from docparser_feb.scripts.nlp.split_text import split_text_by_words_num
from docparser_feb.scripts.openai.prompts import PROMPT_FORMAT, PROMPT_FORMAT_HIS


log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.getLogger().setLevel(LOGING_MAP[log_level])


token = os.getenv("DOC_PARSER_OPENAI_KEY", "")
url = os.getenv("DOC_PARSER_OPENAI_URL", "")
retry_time = int(os.getenv("DOC_PARSER_OPENAI_RETRY", 0))
model_type = os.getenv("DOC_PARSER_OPENAI_MODEL", "")


def get_response(payload):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    idx = 0
    while idx < retry_time:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error(f"OPENAI formatting error: {e}")
            time.sleep(5)
            idx += 1
    return None


def get_format_data(prompt):
    payload = {
        "model": model_type,
        "temperature": 0.01,
        "top_p": 0.4,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a text corrector."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 4096,
        "n": 1,
    }
    corrected_data = get_response(payload)
    if (
        corrected_data is not None
        and "[BEGIN]" in corrected_data
        and "[END]" in corrected_data
    ):
        corrected_data = (
            corrected_data.replace("[BEGIN]", "")
            .replace("[END]", "")
            .strip()
            .strip("`")
            .strip()
            .strip("\n")
            .strip(" ")
        )
        return corrected_data
    else:
        return None


def format_data(text: str):
    split_texts: list = split_text_by_words_num(text)
    new_split_text = []
    for idx, sub_text in enumerate(
        tqdm(split_texts, desc="begin format text by llm", leave=False, position=0)
    ):
        if idx == 0:
            prompt = PROMPT_FORMAT.format(data=sub_text)
        else:
            history_text_str = "\n\n".join(new_split_text[max(0, idx - 2) :])
            prompt = PROMPT_FORMAT_HIS.format(last_data=history_text_str, data=sub_text)

        new_sub_text = get_format_data(prompt)
        if new_sub_text:
            new_split_text.append(new_sub_text)
        else:
            new_split_text.append(sub_text)
    text = " ".join(new_split_text)
    return text
