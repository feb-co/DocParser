import re
import numpy as np

from src.vision.seeit import draw_box
from src.vision import LayoutRecognizer


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
    assert len(page_images) == len(layout_dict)
    imgs = []
    for key in layout_dict:
        img = draw_box(page_images[int(key)-1], layout_dict[key], labels, 0.0)
        imgs.append(img)
    return imgs
