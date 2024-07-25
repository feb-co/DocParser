import os
import time
import requests

from scripts.openai.prompts import PROMPT_FORMAT, PROMPT_FORMAT_HIS


def get_response(payload):
    token = os.environ.get("DOC_PARSER_OPENAI_KEY")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    url = os.environ.get("DOC_PARSER_OPENAI_URL")
    retry_time = int(os.environ.get("DOC_PARSER_OPENAI_RETRY"))
    idx = 0
    while idx < retry_time:
        try:
            response = requests.post(
                url, headers=headers, json=payload
            )
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            time.sleep(5)
            idx += 1


def format_data(text: str, history_text: list = []):
    if len(history_text) > 0:
        history_text_str = "\n\n".join(history_text)
        prompt = PROMPT_FORMAT_HIS.format(last_data=history_text_str, data=text)
    else:
        prompt = PROMPT_FORMAT.format(data=text)

    payload = {
        "model": os.environ.get("DOC_PARSER_OPENAI_MODEL"),
        "temperature": 0.01,
        "repetition_penalty": 1.2,
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

    return text
