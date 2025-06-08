#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT License.

import os.path
import cv2
import logging
import re
import json
from copy import deepcopy
from typing import List, Iterable
from hydra.utils import to_absolute_path
from ultralytics import SAM
from ultralytics.engine.results import Results
from src.engine.model import Model
from src.engine.model import ModelConfig
from src.engine.task import TaskConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SamModel(Model):
    def __init__(
        self,
        cfg: ModelConfig = None,
        task_cfg: TaskConfig = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.task_cfg = task_cfg
        # self._sub_task_cfg = self._exclude_keys()

        logger.info(f"SAM config: {cfg}")
        logger.info(f"SAM task config: {task_cfg}")

        logger.info("construct model")
        self.model = SAM(model=self.task_cfg.abs_ckpt)

        if self.task_cfg.prompt_type:
            logger.info(f"use prompt '{self.task_cfg.prompt_type}'")
        else:
            logger.info("not use prompt")

    def _exclude_keys(self):
        predict_task_cfg = deepcopy(self.task_cfg)
        for key in self.task_cfg.__dict__:
            if key in self.task_cfg.EXCLUSIVE_KEYS:
                predict_task_cfg.__delattr__(key)
        predict_task_cfg.save = self.task_cfg.save_run

        return predict_task_cfg

    def predict(self, source, *args, **kwargs,):
        if isinstance(source, Iterable):
            _source = source
        elif os.path.isdir(d := to_absolute_path(str(source))):
            _source = os.listdir(d)
        else:
            _source = [str(source), ]

        results = []
        for path in _source:
            abs_path = to_absolute_path(path)
            assert os.path.isfile(abs_path), f"No such file {abs_path}"
            f_name = os.path.splitext(os.path.basename(abs_path))[0]
            points = None
            if self.task_cfg.prompt_type:
                points = self.look_up_points(f_name)
                if points is None:
                    logger.info(f"no point")
                else:
                    logger.info(f"found points")
            result = self.model.predict(
                source=abs_path,
                points=points,
            )[0]
            results.append(result)
            self.postprocess(path, result)

            if self.task_cfg.export_box:
                logger.info(f"export boxes")
                os.makedirs(self.task_cfg.box_dir, exist_ok=True)
                self.write_boxes(result, f_name)

            if self.task_cfg.save:
                logger.info(f"save output")
                os.makedirs(self.task_cfg.project, exist_ok=True)
                if points:
                    f_name = f"{f_name}_{self.task_cfg.prompt_type}.jpg"
                else:
                    f_name = f"{f_name}.jpg"

                img = result.plot(
                    save=self.task_cfg.show_labels,
                    line_width=self.task_cfg.line_width,
                    labels=self.task_cfg.show_labels,
                    boxes=self.task_cfg.show_boxes,
                    masks=self.task_cfg.show_masks,
                )
                cv2.imwrite(os.path.join(self.task_cfg.project, f_name), img)

        return results

    def look_up_points(self, f_name: str) -> List[list]|None:
        prompt_file = os.path.join(self.task_cfg.prompt_dir, f"{f_name}.txt")
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
        boxes_file = os.path.join(self.task_cfg.box_dir, f"{f_name}.txt")
        with open(boxes_file, "w") as f:
            for x, y, w, h in result.boxes.xywhn:
                f.write(f"0 {x} {y} {w} {h}\n")

    def train(self, *args, **kwargs):
        raise NotImplemented("Not support for training")
