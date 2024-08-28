import os
import re
import torch
from tqdm import tqdm
from langdetect import detect, LangDetectException

import xgboost as xgb
from io import BytesIO

import pdfplumber
import logging
from PIL import Image
import numpy as np
from PyPDF2 import PdfReader as pdf2_read
from copy import deepcopy
from huggingface_hub import snapshot_download

from docparser_feb.src.vision import (
    OCR,
    Nougat,
    Recognizer,
    LayoutRecognizer,
    TableStructureRecognizer,
)

from docparser_feb.scripts.log_level import LOGING_MAP
from docparser_feb.scripts.file_utils import get_project_base_directory
from docparser_feb.scripts.nlp import rag_tokenizer

log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.getLogger().setLevel(LOGING_MAP[log_level])


class PdfParser:
    def __init__(self):
        self.ocr = OCR()
        self.nougat = Nougat()
        if hasattr(self, "model_speciess"):
            self.layouter = LayoutRecognizer("layout." + self.model_speciess)
        else:
            self.layouter = LayoutRecognizer("layout")
        self.tbl_det = TableStructureRecognizer()

        self.updown_cnt_mdl = xgb.Booster()
        if torch.cuda.is_available():
            self.updown_cnt_mdl.set_param({"device": "cuda"})
        try:
            model_dir = os.path.join(
                get_project_base_directory(), os.environ.get("DOC_PARSER_MODEL_DIR")
            )
            self.updown_cnt_mdl.load_model(
                os.path.join(model_dir, "updown_concat_xgb.model")
            )
        except Exception as e:
            model_dir = snapshot_download(
                repo_id="InfiniFlow/text_concat_xgb_v1.0",
                local_dir=os.path.join(
                    get_project_base_directory(), os.environ.get("DOC_PARSER_MODEL_DIR")
                ),
                local_dir_use_symlinks=False,
            )
            self.updown_cnt_mdl.load_model(
                os.path.join(model_dir, "updown_concat_xgb.model")
            )

        self.page_from = 0

    def __char_width(self, c):
        return (c["x1"] - c["x0"]) // max(len(c["text"]), 1)

    def __height(self, c):
        return c["bottom"] - c["top"]

    def _x_dis(self, a, b):
        return min(
            abs(a["x1"] - b["x0"]),
            abs(a["x0"] - b["x1"]),
            abs(a["x0"] + a["x1"] - b["x0"] - b["x1"]) / 2,
        )

    def _y_dis(self, a, b):
        return (b["top"] + b["bottom"] - a["top"] - a["bottom"]) / 2

    def _match_proj(self, b):
        proj_patt = [
            r"第[零一二三四五六七八九十百]+章",
            r"第[零一二三四五六七八九十百]+[条节]",
            r"[零一二三四五六七八九十百]+[、是 　]",
            r"[\(（][零一二三四五六七八九十百]+[）\)]",
            r"[\(（][0-9]+[）\)]",
            r"[0-9]+(、|\.[　 ]|）|\.[^0-9./a-zA-Z_%><-]{4,})",
            r"[0-9]+\.[0-9.]+(、|\.[ 　])",
            r"[⚫•➢①② ]",
        ]
        return any([re.match(p, b["text"]) for p in proj_patt])

    def _updown_concat_features(self, up, down):
        w = max(self.__char_width(up), self.__char_width(down))
        h = max(self.__height(up), self.__height(down))
        y_dis = self._y_dis(up, down)
        LEN = 6
        tks_down = rag_tokenizer.tokenize(down["text"][:LEN]).split(" ")
        tks_up = rag_tokenizer.tokenize(up["text"][-LEN:]).split(" ")
        tks_all = (
            up["text"][-LEN:].strip()
            + (
                " "
                if re.match(r"[a-zA-Z0-9]+", up["text"][-1] + down["text"][0])
                else ""
            )
            + down["text"][:LEN].strip()
        )
        tks_all = rag_tokenizer.tokenize(tks_all).split(" ")
        fea = [
            up.get("R", -1) == down.get("R", -1),
            y_dis / h,
            down["page_number"] - up["page_number"],
            up["layout_type"] == down["layout_type"],
            up["layout_type"] == "text",
            down["layout_type"] == "text",
            up["layout_type"] == "table",
            down["layout_type"] == "table",
            True if re.search(r"([。？！；!?;+)）]|[a-z]\.)$", up["text"]) else False,
            True if re.search(r"[，：‘“、0-9（+-]$", up["text"]) else False,
            (
                True
                if re.search(r"(^.?[/,?;:\]，。；：’”？！》】）-])", down["text"])
                else False
            ),
            True if re.match(r"[\(（][^\(\)（）]+[）\)]$", up["text"]) else False,
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            (
                True
                if re.search(r"[\(（][^\)）]+$", up["text"])
                and re.search(r"[\)）]", down["text"])
                else False
            ),
            self._match_proj(down),
            True if re.match(r"[A-Z]", down["text"]) else False,
            True if re.match(r"[A-Z]", up["text"][-1]) else False,
            True if re.match(r"[a-z0-9]", up["text"][-1]) else False,
            True if re.match(r"[0-9.%,-]+$", down["text"]) else False,
            (
                up["text"].strip()[-2:] == down["text"].strip()[-2:]
                if len(up["text"].strip()) > 1 and len(down["text"].strip()) > 1
                else False
            ),
            up["x0"] > down["x1"],
            abs(self.__height(up) - self.__height(down))
            / min(self.__height(up), self.__height(down)),
            self._x_dis(up, down) / max(w, 0.000001),
            (len(up["text"]) - len(down["text"]))
            / max(len(up["text"]), len(down["text"])),
            len(tks_all) - len(tks_up) - len(tks_down),
            len(tks_down) - len(tks_up),
            tks_down[-1] == tks_up[-1],
            max(down["in_row"], up["in_row"]),
            abs(down["in_row"] - up["in_row"]),
            len(tks_down) == 1 and rag_tokenizer.tag(tks_down[0]).find("n") >= 0,
            len(tks_up) == 1 and rag_tokenizer.tag(tks_up[0]).find("n") >= 0,
        ]
        return fea

    @staticmethod
    def sort_X_by_page(arr, threashold):
        # sort using y1 first and then x1
        arr = sorted(arr, key=lambda r: (r["page_number"], r["x0"], r["top"]))
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                # restore the order using th
                if (
                    abs(arr[j + 1]["x0"] - arr[j]["x0"]) < threashold
                    and arr[j + 1]["top"] < arr[j]["top"]
                    and arr[j + 1]["page_number"] == arr[j]["page_number"]
                ):
                    tmp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = tmp
        return arr

    def _has_color(self, o):
        if o.get("ncs", "") == "DeviceGray":
            if (
                o["stroking_color"]
                and o["stroking_color"][0] == 1
                and o["non_stroking_color"]
                and o["non_stroking_color"][0] == 1
            ):
                if re.match(r"[a-zT_\[\]\(\)-]+", o.get("text", "")):
                    return False
        return True

    def __table_transformer_job(self, table_bxs, table_layout):
        def gather(kwd, fzy=10, ption=0.6):
            eles = Recognizer.sort_Y_firstly(
                [r for r in table_layout if re.match(kwd, r["label"])], fzy
            )
            eles = Recognizer.layouts_cleanup(table_bxs, eles, 5, ption)
            return Recognizer.sort_Y_firstly(eles, 0)

        # add R,H,C,SP tag to boxes within table layout
        headers = gather(r".*header$")
        rows = gather(r".* (row|header)")
        spans = gather(r".*spanning")
        clmns = sorted(
            [r for r in table_layout if re.match(r"table column$", r["label"])],
            key=lambda x: (x["pn"], x["x0"]),
        )
        clmns = Recognizer.layouts_cleanup(table_bxs, clmns, 5, 0.5)
        for b in table_bxs:
            ii = Recognizer.find_overlapped_with_threashold(b, rows, thr=0.3)
            if ii is not None:
                b["R"] = ii
                b["R_top"] = rows[ii]["top"]
                b["R_bott"] = rows[ii]["bottom"]

            ii = Recognizer.find_overlapped_with_threashold(b, headers, thr=0.3)
            if ii is not None:
                b["H_top"] = headers[ii]["top"]
                b["H_bott"] = headers[ii]["bottom"]
                b["H_left"] = headers[ii]["x0"]
                b["H_right"] = headers[ii]["x1"]
                b["H"] = ii

            ii = Recognizer.find_horizontally_tightest_fit(b, clmns)
            if ii is not None:
                b["C"] = ii
                b["C_left"] = clmns[ii]["x0"]
                b["C_right"] = clmns[ii]["x1"]

            ii = Recognizer.find_overlapped_with_threashold(b, spans, thr=0.3)
            if ii is not None:
                b["H_top"] = spans[ii]["top"]
                b["H_bott"] = spans[ii]["bottom"]
                b["H_left"] = spans[ii]["x0"]
                b["H_right"] = spans[ii]["x1"]
                b["SP"] = ii

        return table_bxs

    def __ocr(self, pagenum, img, chars, ZM=3):
        bxs = self.ocr.detect(np.array(img))

        if not bxs:
            self.boxes.append([])
            return

        bxs = [(line[0], line[1][0]) for line in bxs]
        bxs = Recognizer.sort_Y_firstly(
            [
                {
                    "x0": b[0][0] / ZM,
                    "x1": b[1][0] / ZM,
                    "top": b[0][1] / ZM,
                    "text": "",
                    "txt": t,
                    "bottom": b[-1][1] / ZM,
                    "page_number": pagenum,
                }
                for b, t in bxs
                if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
            ],
            self.mean_height[-1] / 3,
        )

        # merge chars in the same rect
        for c in Recognizer.sort_X_firstly(chars, self.mean_width[pagenum - 1] // 4):
            ii = Recognizer.find_overlapped(c, bxs)
            if ii is None:
                self.lefted_chars.append(c)
                continue
            ch = c["bottom"] - c["top"]
            bh = bxs[ii]["bottom"] - bxs[ii]["top"]
            if abs(ch - bh) / max(ch, bh) >= 0.7 and c["text"] != " ":
                self.lefted_chars.append(c)
                continue
            if c["text"] == " " and bxs[ii]["text"]:
                if re.match(r"[0-9a-zA-Z,.?;:!%%]", bxs[ii]["text"][-1]):
                    bxs[ii]["text"] += " "
            else:
                bxs[ii]["text"] += c["text"]

        for b in bxs:
            # if not b["text"]:
            left, right, top, bott = (
                b["x0"] * ZM,
                b["x1"] * ZM,
                b["top"] * ZM,
                b["bottom"] * ZM,
            )
            b["x0"] *= ZM
            b["x1"] *= ZM
            b["top"] *= ZM
            b["bottom"] *= ZM
            b["text"] = self.ocr.recognize(
                np.array(img),
                np.array(
                    [[left, top], [right, top], [right, bott], [left, bott]],
                    dtype=np.float32,
                ),
            )
            del b["txt"]
        bxs = [b for b in bxs if b["text"]]
        if self.mean_height[-1] == 0:
            self.mean_height[-1] = np.median([b["bottom"] - b["top"] for b in bxs])
        self.boxes.append(bxs)

    def _layouts_rec(self, drop=True):
        assert len(self.page_images) == len(self.boxes)

        self.boxes, self.page_layout = self.layouter(
            self.page_images, self.boxes, drop=drop
        )

    def _text_merge(self):
        # merge adjusted boxes
        bxs_per_page = {}
        for box in self.boxes:
            if box["page_number"] not in bxs_per_page:
                bxs_per_page[box["page_number"]] = []
            bxs_per_page[box["page_number"]].append(box)

        # horizontally merge adjacent box with the same layout
        boxes = []
        for pn in bxs_per_page:
            bxs = bxs_per_page[pn]
            i = 0
            while i < len(bxs) - 1:
                bxs_c = bxs[i]
                bxs_next = bxs[i + 1]

                if bxs_c.get("layoutno", "0") != bxs_next.get(
                    "layoutno", "1"
                ) or bxs_c.get("layout_type", "") in ["table", "figure", "equation"]:
                    i += 1
                    continue

                if (
                    abs(self._y_dis(bxs_c, bxs_next))
                    < self.mean_height[bxs[i]["page_number"] - 1] / 3
                ):
                    # merge
                    # bxs[i]["x1"] = bxs_next["x1"]
                    # bxs[i]["top"] = (bxs_c["top"] + bxs_next["top"]) / 2
                    # bxs[i]["bottom"] = (bxs_c["bottom"] + bxs_next["bottom"]) / 2

                    bxs[i]["x0"] = min(bxs_c["x0"], bxs_next["x0"])
                    bxs[i]["x1"] = max(bxs_c["x1"], bxs_next["x1"])
                    bxs[i]["top"] = min(bxs_c["top"], bxs_next["top"])
                    bxs[i]["bottom"] = max(bxs_c["bottom"], bxs_next["bottom"])
                    bxs[i]["text"] += bxs_next["text"]
                    bxs.pop(i + 1)
                    continue

                i += 1
                continue

            boxes += bxs

        self.boxes = boxes

    def _naive_vertical_merge(self):
        bxs_per_page = {}
        for box in self.boxes:
            if box["page_number"] not in bxs_per_page:
                bxs_per_page[box["page_number"]] = []
            bxs_per_page[box["page_number"]].append(box)

        boxes = []
        for pn in bxs_per_page:
            bxs = bxs_per_page[pn]

            # count boxes in the same row as a feature
            for i in range(len(bxs)):
                mh = self.mean_height[bxs[i]["page_number"] - 1]
                bxs[i]["in_row"] = 0
                j = max(0, i - 12)
                while j < min(i + 12, len(bxs)):
                    if j == i:
                        j += 1
                        continue
                    ydis = self._y_dis(bxs[i], bxs[j]) / mh
                    if abs(ydis) < 1:
                        bxs[i]["in_row"] += 1
                    elif ydis > 0:
                        break
                    j += 1

            bxs = Recognizer.sort_Y_firstly(bxs, np.median(self.mean_height) / 3)
            i = 0
            while i + 1 < len(bxs):
                bxs_c = bxs[i]
                bxs_next = bxs[i + 1]

                # is invalid character
                if bxs_c["page_number"] < bxs_next["page_number"] and re.match(
                    r"[0-9  •一—-]+$", bxs_next["text"]
                ):
                    bxs.pop(i)
                    continue

                if not bxs_c["text"].strip() and bxs_c["layout_type"] in ("text",):
                    bxs.pop(i)
                    continue

                if (
                    "layout_type" in bxs_c
                    and (
                        bxs_c["layout_type"] == "figure caption"
                        or bxs_c["layout_type"] == ""
                    )
                    and "layout_type" in bxs_next
                    and (
                        bxs_next["layout_type"] == "figure caption"
                        or bxs_next["layout_type"] == ""
                    )
                ):
                    if (
                        ("," in bxs_c["text"] or "，" in bxs_c["text"])
                        and abs(bxs_c["bottom"] - bxs_next["bottom"]) > 1
                        and abs(bxs_c["x0"] - bxs_next["x0"]) > 1.5
                    ):
                        bxs_c["layout_type"] = "text"

                if (
                    "layout_type" in bxs_c
                    and bxs_c["layout_type"] == "figure"
                    and "layout_type" in bxs_next
                    and bxs_next["layout_type"] == "figure caption"
                ):
                    if ("," in bxs_next["text"] or "，" in bxs_next["text"]) and (
                        ". " in bxs_next["text"] or "。" in bxs_next["text"]
                    ):
                        bxs_next["layout_type"] = "text"

                if (
                    i - 1 >= 0
                    and "layout_type" in bxs[i - 1]
                    and bxs[i - 1]["layout_type"] in ("figure caption", "text")
                    and "layout_type" in bxs_c
                    and bxs_c["layout_type"] == "figure caption"
                    and "layout_type" in bxs_next
                    and bxs_next["layout_type"] == "table caption"
                ):
                    bxs_c["layout_type"] = "text"

                # features for concating
                try:
                    concat_features = self._updown_concat_features(bxs_c, bxs_next)
                except:
                    i += 1
                    continue
                is_concatting = (
                    self.updown_cnt_mdl.predict(xgb.DMatrix([concat_features]))[0] > 0.5
                )

                # features for not concating
                def not_same_text_witdth():
                    is_same_witdth = (
                        abs(bxs_c["x0"] - bxs_next["x0"]) < 1.0
                        and abs(bxs_c["x1"] - bxs_next["x1"]) < 1.0
                    )

                    try:
                        if (
                            bxs_c["text"].strip()[-1] in "。？！?"
                            and not is_same_witdth
                        ):
                            return True
                        elif bxs_c["text"].strip()[-1] in ".!?" and not is_same_witdth:
                            return True
                        else:
                            return False
                    except:
                        return False

                is_not_concatting = [
                    (
                        "layoutno" in bxs_c
                        and "layoutno" in bxs_next
                        and bxs_c.get("layoutno", 0) != bxs_next.get("layoutno", 0)
                    ),
                    not_same_text_witdth(),
                    bxs_c["page_number"] == bxs_next["page_number"]
                    and bxs_next["top"] - bxs_c["bottom"]
                    > self.mean_height[bxs_next["page_number"] - 1] * 1.5,
                    bxs_c["page_number"] < bxs_next["page_number"]
                    and abs(bxs_c["x0"] - bxs_next["x0"])
                    > self.mean_width[bxs_c["page_number"] - 1] * 4,
                ]

                # split features
                detach_feats = [
                    bxs_c["x1"] < bxs_next["x0"],
                    bxs_c["x0"] > bxs_next["x1"],
                ]

                if (any(is_not_concatting) and not is_concatting) or any(detach_feats):
                    i += 1
                    continue

                # merge up and down
                bxs_c["bottom"] = bxs_next["bottom"]
                bxs_c["text"] = bxs_c["text"] + " " + bxs_next["text"]
                bxs_c["x0"] = min(bxs_c["x0"], bxs_next["x0"])
                bxs_c["x1"] = max(bxs_c["x1"], bxs_next["x1"])
                bxs.pop(i + 1)

            boxes += bxs

        self.boxes = boxes

    def _concat_downward(self, concat_between_pages=True):
        # count boxes in the same row as a feature
        for i in range(len(self.boxes)):
            mh = self.mean_height[self.boxes[i]["page_number"] - 1]
            self.boxes[i]["in_row"] = 0
            j = max(0, i - 12)
            while j < min(i + 12, len(self.boxes)):
                if j == i:
                    j += 1
                    continue
                ydis = self._y_dis(self.boxes[i], self.boxes[j]) / mh
                if abs(ydis) < 1:
                    self.boxes[i]["in_row"] += 1
                elif ydis > 0:
                    break
                j += 1

        # concat between rows
        boxes = deepcopy(self.boxes)
        blocks = []
        while boxes:
            chunks = []

            def dfs(up, dp):
                chunks.append(up)
                i = dp
                while i < min(dp + 12, len(boxes)):
                    ydis = self._y_dis(up, boxes[i])
                    smpg = up["page_number"] == boxes[i]["page_number"]
                    mh = self.mean_height[up["page_number"] - 1]
                    mw = self.mean_width[up["page_number"] - 1]
                    if smpg and ydis > mh * 4:
                        break
                    if not smpg and ydis > mh * 16:
                        break
                    down = boxes[i]
                    if (
                        not concat_between_pages
                        and down["page_number"] > up["page_number"]
                    ):
                        break

                    if up.get("R", "") != down.get("R", "") and up["text"][-1] != "，":
                        i += 1
                        continue

                    if (
                        re.match(r"[0-9]{2,3}/[0-9]{3}$", up["text"])
                        or re.match(r"[0-9]{2,3}/[0-9]{3}$", down["text"])
                        or not down["text"].strip()
                    ):
                        i += 1
                        continue

                    if not down["text"].strip():
                        i += 1
                        continue

                    if (
                        up["x1"] < down["x0"] - 10 * mw
                        or up["x0"] > down["x1"] + 10 * mw
                    ):
                        i += 1
                        continue

                    if i - dp < 5 and up.get("layout_type") == "text":
                        if up.get("layoutno", "1") == down.get("layoutno", "2"):
                            dfs(down, i + 1)
                            boxes.pop(i)
                            return
                        i += 1
                        continue

                    fea = self._updown_concat_features(up, down)
                    if self.updown_cnt_mdl.predict(xgb.DMatrix([fea]))[0] <= 0.5:
                        i += 1
                        continue
                    dfs(down, i + 1)
                    boxes.pop(i)
                    return

            dfs(boxes[0], 1)
            boxes.pop(0)
            if chunks:
                blocks.append(chunks)

        # concat within each block
        boxes = []
        for b in blocks:
            if len(b) == 1:
                boxes.append(b[0])
                continue
            t = b[0]
            for c in b[1:]:
                t["text"] = t["text"].strip()
                c["text"] = c["text"].strip()
                if not c["text"]:
                    continue
                if t["text"] and re.match(
                    r"[0-9\.a-zA-Z]+$", t["text"][-1] + c["text"][-1]
                ):
                    t["text"] += " "
                t["text"] += c["text"]
                t["x0"] = min(t["x0"], c["x0"])
                t["x1"] = max(t["x1"], c["x1"])
                t["page_number"] = min(t["page_number"], c["page_number"])
                t["bottom"] = c["bottom"]
                if not t["layout_type"] and c["layout_type"]:
                    t["layout_type"] = c["layout_type"]
            boxes.append(t)

        self.boxes = Recognizer.sort_Y_firstly(boxes, 0)

    def _filter_forpages(self):
        if not self.boxes:
            return

        findit = False
        i = 0
        while i < len(self.boxes):
            if not re.match(
                r"(contents|目录|目次|table of contents|致谢|acknowledge)$",
                re.sub(r"( | |\u3000)+", "", self.boxes[i]["text"].lower()),
            ):
                i += 1
                continue

            findit = True
            eng = re.match(r"[0-9a-zA-Z :'.-]{5,}", self.boxes[i]["text"].strip())
            self.boxes.pop(i)
            if i >= len(self.boxes):
                break

            prefix = (
                self.boxes[i]["text"].strip()[:3]
                if not eng
                else " ".join(self.boxes[i]["text"].strip().split(" ")[:2])
            )
            while not prefix:
                self.boxes.pop(i)
                if i >= len(self.boxes):
                    break
                prefix = (
                    self.boxes[i]["text"].strip()[:3]
                    if not eng
                    else " ".join(self.boxes[i]["text"].strip().split(" ")[:2])
                )

            self.boxes.pop(i)
            if i >= len(self.boxes) or not prefix:
                break

            for j in range(i, min(i + 128, len(self.boxes))):
                try:
                    if not re.match(prefix, self.boxes[j]["text"]):
                        continue
                except:
                    continue
                for k in range(i, j):
                    self.boxes.pop(i)
                break

        if findit:
            return

        page_dirty = [0] * len(self.page_images)
        for b in self.boxes:
            if re.search(r"(··|··|··)", b["text"]):
                page_dirty[b["page_number"] - 1] += 1
        page_dirty = set([i + 1 for i, t in enumerate(page_dirty) if t > 3])
        if not page_dirty:
            return

        i = 0
        while i < len(self.boxes):
            if self.boxes[i]["page_number"] in page_dirty:
                self.boxes.pop(i)
                continue
            i += 1

    def _merge_with_same_bullet(self):
        i = 0
        while i + 1 < len(self.boxes):
            bxs_c = self.boxes[i]
            bxs_next = self.boxes[i + 1]

            if not bxs_c["text"].strip() and bxs_c["layout_type"] in ("text"):
                self.boxes.pop(i)
                continue

            if not bxs_next["text"].strip() and bxs_next["layout_type"] in ("text"):
                self.boxes.pop(i + 1)
                continue

            if (
                not bxs_c["text"].strip()
                or not bxs_next["text"].strip()
                or bxs_c["text"].strip()[0] != bxs_next["text"].strip()[0]
                or bxs_c["text"].strip()[0].lower() in set("qwertyuopasdfghjklzxcvbnm")
                or rag_tokenizer.is_chinese(bxs_c["text"].strip()[0])
                or bxs_c["top"] > bxs_next["bottom"]
            ):
                i += 1
                continue

            bxs_next["text"] = bxs_c["text"] + "\n" + bxs_next["text"]
            bxs_next["x0"] = min(bxs_c["x0"], bxs_next["x0"])
            bxs_next["x1"] = max(bxs_c["x1"], bxs_next["x1"])
            bxs_next["top"] = bxs_c["top"]
            self.boxes.pop(i)

    def _extract_table_figure(self, return_html):
        tables = {}
        figures = {}

        # extract figure and table boxes
        i = 0
        lst_lout_no = ""
        nomerge_lout_no = []
        while i < len(self.boxes):
            if "layoutno" not in self.boxes[i]:
                i += 1
                continue

            lout_no = (
                str(self.boxes[i]["page_number"]) + "-" + str(self.boxes[i]["layoutno"])
            )

            if TableStructureRecognizer.is_caption(self.boxes[i]) or self.boxes[i][
                "layout_type"
            ] in ["table caption", "title", "figure caption", "reference"]:
                nomerge_lout_no.append(lst_lout_no)

            if self.boxes[i]["layout_type"] == "table":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in tables:
                    tables[lout_no] = []
                tables[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue

            if self.boxes[i]["layout_type"] == "figure":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in figures:
                    figures[lout_no] = []
                figures[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            i += 1

        # merge table on different pages
        nomerge_lout_no = set(nomerge_lout_no)
        tbls = sorted(
            [(k, bxs) for k, bxs in tables.items()],
            key=lambda x: (x[1][0]["top"], x[1][0]["x0"]),
        )

        i = len(tbls) - 1
        while i - 1 >= 0:
            k0, bxs0 = tbls[i - 1]
            k, bxs = tbls[i]
            i -= 1
            if k0 in nomerge_lout_no:
                continue
            if bxs[0]["page_number"] == bxs0[0]["page_number"]:
                continue
            if bxs[0]["page_number"] - bxs0[0]["page_number"] > 1:
                continue
            mh = self.mean_height[bxs[0]["page_number"] - 1]
            if self._y_dis(bxs0[-1], bxs[0]) > mh * 23:
                continue
            tables[k0].extend(tables[k])
            del tables[k]

        def x_overlapped(a, b):
            return not any([a["x1"] < b["x0"], a["x0"] > b["x1"]])

        # find captions and pop out
        i = 0
        while i < len(self.boxes):
            c_box = self.boxes[i]
            # mh = self.mean_height[c["page_number"]-1]
            if not TableStructureRecognizer.is_caption(c_box):
                i += 1
                continue

            # find the nearest layouts
            def nearest(tbls):
                nonlocal c_box
                mink = ""
                minv = 1000000000
                for k, bxs in tbls.items():
                    for b in bxs:
                        if (
                            b.get("layout_type", "").find("caption") >= 0
                            or b["page_number"] != c_box["page_number"]
                        ):
                            continue
                        y_dis = self._y_dis(c_box, b)
                        x_dis = (
                            self._x_dis(c_box, b) if not x_overlapped(c_box, b) else 0
                        )
                        dis = y_dis * y_dis + x_dis * x_dis
                        if dis < minv:
                            mink = k
                            minv = dis
                return mink, minv

            tk, tv = nearest(tables)
            fk, fv = nearest(figures)
            # if min(tv, fv) > 2000:
            #    i += 1
            #    continue
            if tv < fv and tk:
                tables[tk].insert(0, c_box)
                logging.debug("TABLE:" + self.boxes[i]["text"] + "; Cap: " + tk)
            elif fk:
                figures[fk].insert(0, c_box)
                logging.debug("FIGURE:" + self.boxes[i]["text"] + "; Cap: " + tk)
            self.boxes.pop(i)

        res = []

        def cropout(bxs, ltype, poss):
            pn = set([b["page_number"] for b in bxs])
            if len(pn) < 2:
                pn = list(pn)[0]
                b = {
                    "x0": np.min([b["x0"] for b in bxs]),
                    "top": np.min([b["top"] for b in bxs]),
                    "x1": np.max([b["x1"] for b in bxs]),
                    "bottom": np.max([b["bottom"] for b in bxs]),
                }
                left, top, right, bott = b["x0"], b["top"], b["x1"], b["bottom"]
                if right < left:
                    right = left + 1
                poss.append((pn, left, right, top, bott))
                new_img = self.page_images[pn - 1].crop((left, top, right, bott))
                return new_img

            pn = {}
            for b in bxs:
                p = b["page_number"] - 1
                if p not in pn:
                    pn[p] = []
                pn[p].append(b)
            pn = sorted(pn.items(), key=lambda x: x[0])
            imgs = [cropout(arr, ltype, poss) for p, arr in pn]
            pic = Image.new(
                "RGB",
                (
                    int(np.max([i.size[0] for i in imgs])),
                    int(np.sum([m.size[1] for m in imgs])),
                ),
                (245, 245, 245),
            )
            height = 0
            for img in imgs:
                pic.paste(img, (0, int(height)))
                height += img.size[1]
            return pic

        # crop figure out and add caption
        logging.debug("figure processing...")
        for k, bxs in figures.items():
            txt = "\n".join([b["text"] for b in bxs])
            if not txt:
                continue

            poss = []
            crop_image = cropout(bxs, "figure", poss)
            res.append(
                {
                    "x0": poss[0][1],
                    "x1": poss[0][2],
                    "top": poss[0][3],
                    "bottom": poss[0][4],
                    "text": txt,
                    "page_number": int(k.split("-")[0]),
                    "layout_type": "figure",
                    "layoutno": k,
                    "image": crop_image,
                }
            )

        # crop table out and add caption
        logging.debug("Table processing...")

        def extract_table_from_img(img, start_x, start_y, page_number):
            bxs = self.ocr.detect(np.array(img))
            if not bxs:
                return None

            bxs = [(line[0], line[1][0]) for line in bxs]
            bxs = Recognizer.sort_Y_firstly(
                [
                    {
                        "x0": b[0][0],
                        "x1": b[1][0],
                        "top": b[0][1],
                        "bottom": b[-1][1],
                        "text": "",
                        "txt": t,
                        "page_number": page_number,
                    }
                    for b, t in bxs
                    if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
                ],
                self.mean_height[-1] / 3,
            )

            for b in bxs:
                # if not b["text"]:
                left, right, top, bott = (
                    b["x0"],
                    b["x1"],
                    b["top"],
                    b["bottom"],
                )
                b["text"] = self.ocr.recognize(
                    np.array(img),
                    np.array(
                        [[left, top], [right, top], [right, bott], [left, bott]],
                        dtype=np.float32,
                    ),
                )
                del b["txt"]

            bxs, _ = self.layouter([img], [bxs], drop=True, update_pos=False)
            bxs = [b for b in bxs if b["text"]]
            for b in bxs:
                b["x0"] += start_x
                b["x1"] += start_x
                b["top"] += start_y
                b["bottom"] += start_y
                if b["layout_type"] == "figure":
                    b["layout_type"] = "table"
                elif "caption" in b["layout_type"]:
                    b["layout_type"] = "table caption"
                else:
                    b["layout_type"] = "table"

                del b["layoutno"]

            # table layout
            table_layout = []
            retry = 0
            while True:
                try:
                    table_layout = self.tbl_det([img])[0]
                    break
                except:
                    if retry > 5:
                        break
                    retry += 1
                    pass
            for item in table_layout:
                item["x0"] += start_x
                item["x1"] += start_x
                item["top"] += start_y
                item["bottom"] += start_y
                item["pn"] = page_number

            return bxs, table_layout

        for k, bxs in tables.items():
            if not bxs:
                continue
            bxs = Recognizer.sort_Y_firstly(
                bxs, np.mean([(b["bottom"] - b["top"]) / 2 for b in bxs])
            )
            poss = []
            crop_image = cropout(bxs, "table", poss)
            try:
                res.append(
                    {
                        "x0": poss[0][1],
                        "x1": poss[0][2],
                        "top": poss[0][3],
                        "bottom": poss[0][4],
                        "text": self.tbl_det.construct_table(
                            self.__table_transformer_job(
                                *extract_table_from_img(
                                    crop_image,
                                    start_x=poss[0][1],
                                    start_y=poss[0][3],
                                    page_number=int(k.split("-")[0]),
                                )
                            ),
                            html=return_html,
                            is_english=self.is_english,
                        ),
                        "page_number": int(k.split("-")[0]),
                        "layout_type": "table",
                        "layoutno": k,
                    }
                )
            except:
                pass

        return res

    def proj_match(self, line):
        if len(line) <= 2:
            return
        if re.match(r"[0-9 ().,%%+/-]+$", line):
            return False
        for p, j in [
            (r"第[零一二三四五六七八九十百]+章", 1),
            (r"第[零一二三四五六七八九十百]+[条节]", 2),
            (r"[零一二三四五六七八九十百]+[、 　]", 3),
            (r"[\(（][零一二三四五六七八九十百]+[）\)]", 4),
            (r"[0-9]+(、|\.[　 ]|\.[^0-9])", 5),
            (r"[0-9]+\.[0-9]+(、|[. 　]|[^0-9])", 6),
            (r"[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 7),
            (r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 8),
            (r".{,48}[：:?？]$", 9),
            (r"[0-9]+）", 10),
            (r"[\(（][0-9]+[）\)]", 11),
            (r"[零一二三四五六七八九十百]+是", 12),
            (r"[⚫•➢✓]", 12),
        ]:
            if re.match(p, line):
                return j
        return

    def _line_tag(self, bx, zoomin):
        pn = [bx["page_number"]]
        top = bx["top"]
        bott = bx["bottom"]
        page_images_cnt = len(self.page_images)
        if pn[-1] - 1 >= page_images_cnt:
            return ""
        while bott * zoomin > self.page_images[pn[-1] - 1].size[1]:
            bott -= self.page_images[pn[-1] - 1].size[1] / zoomin
            pn.append(pn[-1] + 1)
            if pn[-1] - 1 >= page_images_cnt:
                return ""

        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format(
            "-".join([str(p) for p in pn]), bx["x0"], bx["x1"], top, bott
        )

    def _text_predict(
        self,
    ):
        index_list = []
        image_list = []
        MARGIN_VALUE = 10
        for i in range(len(self.boxes)):
            if self.boxes[i]["layout_type"] in ("text", "equation", ""):
                left, top, right, bott = (
                    self.boxes[i]["x0"] - MARGIN_VALUE,
                    self.boxes[i]["top"] - MARGIN_VALUE,
                    self.boxes[i]["x1"] + MARGIN_VALUE,
                    self.boxes[i]["bottom"] + MARGIN_VALUE,
                )
                if right < left:
                    right = left + 1
                image = self.page_images[self.boxes[i]["page_number"] - 1].crop(
                    (left, top, right, bott)
                )
                index_list.append(i)
                image_list.append(image)

        text_list = self.nougat(image_list)
        for i, text in enumerate(text_list):
            if len(text) > 0 and not text.startswith("[MISSING_PAGE"):
                self.boxes[index_list[i]]["text"] = text

    def __filterout_scraps(self, boxes, ZM):

        def width(b):
            return b["x1"] - b["x0"]

        def height(b):
            return b["bottom"] - b["top"]

        def usefull(b):
            if b.get("layout_type"):
                return True
            if width(b) > self.page_images[b["page_number"] - 1].size[0] / ZM / 3:
                return True
            if b["bottom"] - b["top"] > self.mean_height[b["page_number"] - 1]:
                return True
            return False

        res = []
        while boxes:
            lines = []
            widths = []
            pw = self.page_images[boxes[0]["page_number"] - 1].size[0] / ZM
            mh = self.mean_height[boxes[0]["page_number"] - 1]
            mj = (
                self.proj_match(boxes[0]["text"])
                or boxes[0].get("layout_type", "") == "title"
            )

            def dfs(line, st):
                nonlocal mh, pw, lines, widths
                lines.append(line)
                widths.append(width(line))
                width_mean = np.mean(widths)
                mmj = (
                    self.proj_match(line["text"])
                    or line.get("layout_type", "") == "title"
                )
                for i in range(st + 1, min(st + 20, len(boxes))):
                    if (boxes[i]["page_number"] - line["page_number"]) > 0:
                        break
                    if (
                        not mmj
                        and self._y_dis(line, boxes[i]) >= 3 * mh
                        and height(line) < 1.5 * mh
                    ):
                        break

                    if not usefull(boxes[i]):
                        continue
                    if mmj or (
                        self._x_dis(boxes[i], line) < pw / 10
                    ):  # and abs(width(boxes[i])-width_mean)/max(width(boxes[i]),width_mean)<0.5):
                        # concat following
                        dfs(boxes[i], i)
                        boxes.pop(i)
                        break

            try:
                if usefull(boxes[0]):
                    dfs(boxes[0], 0)
                else:
                    logging.debug("WASTE: " + boxes[0]["text"])
            except Exception as e:
                pass
            boxes.pop(0)
            mw = np.mean(widths)
            if mj or mw / pw >= 0.35 or mw > 200:
                res.append(
                    "\n".join([c["text"] + self._line_tag(c, ZM) for c in lines])
                )
            else:
                logging.debug("REMOVED: " + "<<".join([c["text"] for c in lines]))

        return "\n\n".join(res)

    @staticmethod
    def total_page_number(fnm, binary=None):
        try:
            pdf = (
                pdfplumber.open(fnm) if not binary else pdfplumber.open(BytesIO(binary))
            )
            return len(pdf.pages)
        except Exception as e:
            logging.error(str(e))

    def _detect_language(self):
        is_english = []
        for i in range(len(self.boxes)):
            text = "".join(
                [c["text"] for c in self.boxes[i] if len(c["text"].strip()) > 0]
            )
            try:
                # 使用 langdetect 检测语言
                is_english.append(detect(text) == 'en')
            except LangDetectException:
                # 如果检测失败，默认为非英文
                is_english.append(False)
        return is_english

    def __images__(self, fnm, zoomin=3, page_from=0, page_to=299, is_english=None):
        self.lefted_chars = []
        self.mean_height = []
        self.mean_width = []
        self.boxes = []
        self.garbages = {}
        self.page_layout = []
        self.page_from = page_from
        try:
            self.pdf = (
                pdfplumber.open(fnm)
                if isinstance(fnm, str)
                else pdfplumber.open(BytesIO(fnm))
            )
            self.page_images = [
                p.to_image(resolution=72 * zoomin).annotated
                for i, p in enumerate(self.pdf.pages[page_from:page_to])
            ]
            self.page_chars = [
                [c for c in page.chars if self._has_color(c)]
                for page in self.pdf.pages[page_from:page_to]
            ]
            self.total_page = len(self.pdf.pages)
        except Exception as e:
            logging.error(str(e))
            return False

        self.outlines = []
        try:
            self.pdf = pdf2_read(fnm if isinstance(fnm, str) else BytesIO(fnm))
            outlines = self.pdf.outline

            def dfs(arr, depth):
                for a in arr:
                    if isinstance(a, dict):
                        self.outlines.append((a["/Title"], depth))
                        continue
                    dfs(a, depth + 1)

            dfs(outlines, 0)
        except Exception as e:
            logging.warning(f"Outlines exception: {e}")
        if not self.outlines:
            logging.warning(f"Miss outlines")

        logging.debug("Images converted.")

        for i, img in enumerate(tqdm(self.page_images, desc="begin parser pdf pages", leave=False, position=0)):
            chars = []
            self.mean_height.append(
                np.median(sorted([c["height"] for c in chars])) if chars else 0
            )
            self.mean_width.append(
                np.median(sorted([c["width"] for c in chars])) if chars else 8
            )
            j = 0
            while j + 1 < len(chars):
                if (
                    chars[j]["text"]
                    and chars[j + 1]["text"]
                    and re.match(
                        r"[0-9a-zA-Z,.:;!%]+", chars[j]["text"] + chars[j + 1]["text"]
                    )
                    and chars[j + 1]["x0"] - chars[j]["x1"]
                    >= min(chars[j + 1]["width"], chars[j]["width"]) / 2
                ):
                    chars[j]["text"] += " "
                j += 1

            self.__ocr(i + 1, img, chars, zoomin)

        if is_english is None:
            is_english = self._detect_language()
            if (
                sum([1 if e else 0 for e in is_english])
                > len(is_english) / 3
            ):
                self.is_english = True
            else:
                self.is_english = False
                self.boxes = []
                self.mean_height = []
                self.mean_width = []
                for i, img in enumerate(self.page_images):
                    chars = self.page_chars[i]
                    self.mean_height.append(
                        np.median(sorted([c["height"] for c in chars])) if chars else 0
                    )
                    self.mean_width.append(
                        np.median(sorted([c["width"] for c in chars])) if chars else 8
                    )
                    j = 0
                    while j + 1 < len(chars):
                        if (
                            chars[j]["text"]
                            and chars[j + 1]["text"]
                            and re.match(
                                r"[0-9a-zA-Z,.:;!%]+",
                                chars[j]["text"] + chars[j + 1]["text"],
                            )
                            and chars[j + 1]["x0"] - chars[j]["x1"]
                            >= min(chars[j + 1]["width"], chars[j]["width"]) / 2
                        ):
                            chars[j]["text"] += " "
                        j += 1

                    self.__ocr(i + 1, img, chars, zoomin)

            if (
                not self.is_english
                and not any([c for c in self.page_chars])
                and self.boxes
            ):
                bxes = [b for bxs in self.boxes for b in bxs]
                self.is_english = re.search(
                    r"[\na-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}",
                    "".join([b["text"] for b in bxes]),
                )
        else:
            self.is_english = is_english

        logging.debug(f"Is it English: {self.is_english}")

        if len(self.boxes) == 0 and zoomin < 9:
            self.__images__(fnm, zoomin * 3, page_from, page_to)

        return True

    def __call__(self, fnm, need_image=True, zoomin=3, return_html=False):
        self.__images__(fnm, zoomin)
        self._layouts_rec(zoomin)
        self._table_transformer_job(zoomin)
        self._text_merge()
        self._concat_downward()
        self._filter_forpages()
        tbls = self._extract_table_figure(need_image, zoomin, return_html, False)
        return self.__filterout_scraps(deepcopy(self.boxes), zoomin), tbls

    def remove_tag(self, txt):
        return re.sub(r"@@[\t0-9.-]+?##", "", txt)


class PlainParser(object):
    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        self.outlines = []
        lines = []
        try:
            self.pdf = pdf2_read(
                filename if isinstance(filename, str) else BytesIO(filename)
            )
            for page in self.pdf.pages[from_page:to_page]:
                lines.extend([t for t in page.extract_text().split("\n")])

            outlines = self.pdf.outline

            def dfs(arr, depth):
                for a in arr:
                    if isinstance(a, dict):
                        self.outlines.append((a["/Title"], depth))
                        continue
                    dfs(a, depth + 1)

            dfs(outlines, 0)
        except Exception as e:
            logging.warning(f"Outlines exception: {e}")
        if not self.outlines:
            logging.warning(f"Miss outlines")

        return [(l, "") for l in lines], []

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError
