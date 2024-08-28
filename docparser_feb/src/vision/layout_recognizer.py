import os
import re
from collections import Counter
from copy import deepcopy
import numpy as np
import layoutparser as lp
from huggingface_hub import snapshot_download

from docparser_feb.src.vision import Recognizer

from docparser_feb.scripts.file_utils import get_project_base_directory


class LayoutRecognizer(Recognizer):
    labels = [
        "_background_",
        "Text",
        "Title",
        "Figure",
        "Figure caption",
        "Table",
        "Table caption",
        "Header",
        "Footer",
        "Reference",
        "Equation",
    ]

    labels_priority = {
        "Table caption": 0,
        "Table": 1,
        "Figure caption": 2,
        "Title": 3,
        "Header": 4,
        "Footer": 5,
        "Equation": 6,
        "Text": 7,
        "Figure": 8,
        "Reference": 9,
    }

    def __init__(self, domain):
        try:
            model_dir = os.path.join(
                get_project_base_directory(), os.environ.get("DOC_PARSER_MODEL_DIR")
            )
            super().__init__(self.labels, self.labels_priority, domain, model_dir)
        except Exception as e:
            model_dir = snapshot_download(
                repo_id="InfiniFlow/deepdoc",
                local_dir=os.path.join(
                    get_project_base_directory(), os.environ.get("DOC_PARSER_MODEL_DIR")
                ),
                local_dir_use_symlinks=False,
            )
            super().__init__(self.labels, self.labels_priority, domain, model_dir)

        self.garbage_layouts = ["footer", "header", "equation", 'figure']

        self.lp_model = lp.AutoLayoutModel("lp://efficientdet/PubLayNet/tf_efficientdet_d1")
        self.lp_type = ["Figure", "Text"]

    def __call__(self, image_list, ocr_res, thr=0.2, batch_size=16, drop=True, update_pos=True):
        def __is_garbage(b):
            patt = [
                r"^•+$",
                r"(版权归©|免责条款|地址[:：])",
                r"\.{3,}",
                "^[0-9]{1,2} / ?[0-9]{1,2}$",
                r"^[0-9]{1,2} of [0-9]{1,2}$",
                "^http://[^ ]{12,}",
                "(资料|数据)来源[:：]",
                "[0-9a-z._-]+@[a-z0-9-]+\\.[a-z]{2,3}",
                "\\(cid *: *[0-9]+ *\\)",
            ]
            return any([re.search(p, b["text"]) for p in patt])

        layouts = super().__call__(image_list, thr, batch_size)
        assert len(image_list) == len(ocr_res)
        assert len(image_list) == len(layouts)

        for i in range(len(layouts)):
            lp_layout = self.lp_model.detect(image_list[i])
            add_layout = []
            for item in lp_layout._blocks:
                if item.type in self.lp_type:
                    block = {
                        "type": item.type.lower(),
                        "bbox": [
                            item.block.x_1,
                            item.block.y_1,
                            item.block.x_2,
                            item.block.y_2,
                        ],
                        "score": item.score,
                    }
                    ii = self.find_overlapped_with_threashold(
                        {
                            "x0": block["bbox"][0],
                            "x1": block["bbox"][2],
                            "top": block["bbox"][1],
                            "bottom": block["bbox"][-1],
                        },
                        [
                            {
                                "x0": b["bbox"][0],
                                "x1": b["bbox"][2],
                                "top": b["bbox"][1],
                                "bottom": b["bbox"][-1],
                            }
                            for b in layouts[i]
                        ],
                        thr=0.1,
                    )
                    if ii is None:
                        add_layout.append(block)
            layouts[i] += add_layout

        boxes = []
        garbages = {}
        page_layout = []
        for pn, lts in enumerate(layouts):
            bxs = ocr_res[pn]
            lts = [
                {
                    "type": b["type"],
                    "score": float(b["score"]),
                    "x0": b["bbox"][0],
                    "x1": b["bbox"][2],
                    "top": b["bbox"][1],
                    "bottom": b["bbox"][-1],
                    "page_number": pn+1,
                }
                for b in lts
            ]
            lts = self.sort_Y_firstly(
                lts, np.mean([l["bottom"] - l["top"] for l in lts]) / 2
            )
            lts = self.layouts_cleanup(bxs, lts)
            page_layout.append(lts)

            # Tag layout type, layouts are ready
            def findLayout():
                nonlocal bxs, lts, self
                lts_ = [lt for lt in lts]
                i = 0
                while i < len(bxs):
                    if bxs[i].get("layout_type"):
                        i += 1
                        continue

                    if __is_garbage(bxs[i]):
                        bxs.pop(i)
                        continue

                    ii = self.find_overlapped_with_threashold(bxs[i], lts_, thr=0.1)
                    if ii is None:
                        bxs.pop(i)
                        continue

                    lts_[ii]["visited"] = (
                        True if lts_[ii]["type"] not in ("equation", 'figure') else False
                    )

                    keep_feats = [
                        lts_[ii]["type"] == "footer"
                        and bxs[i]["bottom"] < image_list[pn].size[1] * 0.9,
                        lts_[ii]["type"] == "header"
                        and bxs[i]["top"] > image_list[pn].size[1] * 0.1,
                    ]
                    if (
                        drop
                        and lts_[ii]["type"] in self.garbage_layouts
                        and not any(keep_feats)
                    ):
                        if lts_[ii]["type"] not in garbages:
                            garbages[lts_[ii]["type"]] = []
                        garbages[lts_[ii]["type"]].append(bxs[i]["text"])
                        if lts_[ii]["type"] in ["figure", "equation"]:
                            if 'text' not in lts_[ii]:
                                lts_[ii]['text'] = ""
                            lts_[ii]['text'] += bxs[i]['text']
                        bxs.pop(i)
                        continue

                    bxs[i]["layoutno"] = f"{lts_[ii]['type']}-{ii}"
                    bxs[i]["layout_type"] = lts_[ii]["type"]
                    if update_pos:
                        bxs[i]["x0"] = min(lts_[ii]["x0"], bxs[i]["x0"])
                        bxs[i]["x1"] = max(lts_[ii]["x1"], bxs[i]["x1"])
                        bxs[i]["top"] = min(lts_[ii]["top"], bxs[i]["top"])
                        bxs[i]["bottom"] = max(lts_[ii]["bottom"], bxs[i]["bottom"])
                    i += 1

            findLayout()

            for i, lt in enumerate(
                [lt for lt in lts if lt["type"] in ["figure", "equation"]]
            ):
                if lt.get("visited"):
                    continue
                lt = deepcopy(lt)
                lt["text"] = lt["text"] if 'text' in lt else ""
                lt["layout_type"] = lt["type"]
                lt["layoutno"] = f"{lt['type']}-{i}"
                try:
                    del lt["type"]
                except:
                    pass
                try:
                    del lt["visited"]
                except:
                    pass
                try:
                    del lt["score"]
                except:
                    pass
                bxs.append(lt)

            boxes.extend(bxs)
        ocr_res = boxes

        garbag_set = set()
        for k in garbages.keys():
            garbages[k] = Counter(garbages[k])
            for g, c in garbages[k].items():
                if c > 1:
                    garbag_set.add(g)

        ocr_res = [b for b in ocr_res if b["text"].strip() not in garbag_set]
        return ocr_res, page_layout
