#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#
import os.path
import cv2
import logging
import re
import json

from dataclasses import field
from typing import List
from torch.nn import Module
from ultralytics import SAM
from ultralytics.engine.results import Results, Boxes

from src.test_pretrained_model.config import SamConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(__file__))


class SamPredictor(Module):
    config: SamConfig = field(default=SamConfig())
    PROMPT_TYPE: dict = {
        'bbox': "bbox",
        'points': "point",
    }

    def __init__(self, config: SamConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.predictor = SAM(model=self.config.model_path)

    def predict(self, source: str, *args, **kwargs,):
        assert os.path.isfile(source), f"No such file {source}"
        f_name = os.path.splitext(os.path.basename(source))[0]
        points = None
        if self.config.prompt_type:
            logger.info(f"use prompt '{self.config.prompt_type}'")
            points = self.look_up_points(f_name)
        else:
            logger.info("not use prompt")

        results = self.predictor.predict(
            source=source,
            save=self.config.save_run,
            points=points,

        )

        if self.config.export_box:
            logger.info(f"export boxes")
            os.makedirs(self.config.box_dir, exist_ok=True)
            self.write_boxes(results[0], f_name)

        if self.config.save:
            logger.info(f"save output")
            os.makedirs(self.config.project, exist_ok=True)
            if self.config.prompt_type:
                f_name = f"{f_name}_{self.config.prompt_type}.jpg"
            else:
                f_name = f"{f_name}.jpg"

            img = results[0].plot(
                save=self.config.show_labels,
                line_width=self.config.line_width,
                labels=self.config.show_labels,
                boxes=self.config.show_boxes,
                masks=self.config.show_masks,
            )
            cv2.imwrite(os.path.join(self.config.project, f_name),img)

    def look_up_points(self, f_name: str) -> List[list]|None:
        prompt_file = os.path.join(self.config.prompt_dir, f"{f_name}.txt")
        if not os.path.isfile(prompt_file): return None

        with open(prompt_file, mode='r') as f:
            pure_content = re.sub(r'\s*', '', f.read())
            content = re.findall(r"\[{.*}\]", pure_content)[0]

        points = []
        for d in json.loads(content):
            tmp = []
            for k in d:
                tmp.append(d[k])
            points.append(tmp)

        return points

    def write_boxes(self, result: Results, f_name: str):
        boxes_file = os.path.join(self.config.box_dir, f"{f_name}.txt")
        with open(boxes_file, "w") as f:
            for x, y, w, h in result.boxes.xywhn:
                f.write(f"0 {x} {y} {w} {h}\n")
