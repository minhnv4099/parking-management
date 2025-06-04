#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#

import os
import json
import re


def txt_to_json(txt_path: str, json_path: str,):
    with open(txt_path, mode='r') as f:
        content = re.findall(r"\[{.*}\]", string=f.read())[0]
        obj = json.loads(content)

    with open(json_path, mode='w') as f:
        json.dump(obj, indent=4, fp=f)


def json_to_object(json_path: str) -> object:
    if json_path.endswith(".txt"):
        txt_to_json(json_path, json_path)
    with open(json_path, mode='r') as f:
        return json.load(f)
