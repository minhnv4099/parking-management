#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#

from typing import List
from .json import json_to_object


def make_points(json_path: str) -> List:
    obj = json_to_object(json_path)
    entries = []
    for d in obj:
        tmp = []
        for k in d:
            tmp.append(d[k])
        entries.append(tmp)

    return entries

def write_boxes(boxes: List, text_path: str):
    with open(text_path, "w") as f:
        for x, y, w, h in boxes:
            f.write(f"0 {x} {y} {w} {h} \n")

