import re
import numpy as np

from docparser_feb.src.vision.seeit import draw_box
from docparser_feb.src.vision import LayoutRecognizer


def pdf_rendering(page_images, boxes):
    labels = LayoutRecognizer.labels
    labels.append('')

    layout_dict = {}
    for box in boxes:
        if box['page_number'] not in layout_dict:
            layout_dict[box['page_number']] = []
        layout_dict[box['page_number']].append({
            "score": 1.0,
            "type": box.get("layout_type", ''),
            "bbox": [box["x0"], box["top"], box["x1"], box["bottom"]],
        })
    imgs = []
    for key in layout_dict:
        if 0<=int(key)-1<len(page_images):
            img = page_images[int(key)-1]
            try:
                img = draw_box(img, layout_dict[key], labels, 0.0)
                imgs.append(img)
            except:
                imgs.append(img)
        else:
            continue
    return imgs
