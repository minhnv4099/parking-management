#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#

import os
import json
import re


def txt_to_json(txt_path: str) -> str:
    """
    :param txt_path:
    :return: Path to json file
    """
    with open(txt_path, mode='r') as f:
        content = re.findall(r"\[{.*}\]", string=f.read())[0]
        list_points = json.loads(content)

    json_path = txt_path.replace("txt", "json")
    os.makedirs(os.path.split(json_path)[0], exist_ok=True)

    with open(json_path, mode='w') as f:
        json.dump(list_points, indent=4, fp=f)

    return json_path


def json_to_object(json_path: str) -> object:
    if json_path.endswith(".txt"):
        json_path = txt_to_json(json_path)
    with open(json_path, mode='r') as f:
        return json.load(f)