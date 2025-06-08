#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT License.

import os
import logging
from copy import deepcopy
from typing import Iterable, Any
from ultralytics import YOLO
from hydra.utils import to_absolute_path
from src.engine.model import Model, ModelConfig
from src.engine.task import TaskConfig
from src.engine.trainer import Trainer
from src.engine.predictor import Predictor
from src.models.yolo import YoloPredictingConfig, YoloTrainingConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class YoloModel(Model):
    def __init__(
            self,
            cfg: ModelConfig,
            task_cfg: TaskConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.task_cfg = task_cfg
        self._sub_task_cfg = self._exclude_keys()

        logger.info(f"YOLO model config: {cfg}")
        logger.info(f"YOLO task config: {task_cfg}")

        logger.info("construct model")
        self.model = YOLO(
            model=to_absolute_path(self.task_cfg.ckpt),
            task=self.task_cfg.task,
        )

        self.trainer = Trainer(model=self.model)
        self.predictor = Predictor(model=self.model)

    def _exclude_keys(self):
        predict_task_cfg = deepcopy(self.task_cfg)
        for key in self.task_cfg.__dict__:
            if key in self.task_cfg.EXCLUSIVE_KEYS:
                predict_task_cfg.__delattr__(key)
        predict_task_cfg.save = self.task_cfg.save_run

        return predict_task_cfg

    def predict(self, source: Any, *args, **kwargs):
        _source = self.preprocess(source)
        results = []
        for path in _source:
            abs_path = to_absolute_path(path)
            result = self.predictor.predict(abs_path, self._sub_task_cfg)[0]
            results.append(result)
            self.postprocess(path, result)

        return results

    def preprocess(self, source, *args, **kwargs):
        if isinstance(source, Iterable):
            _source = source
        elif os.path.isdir(d := to_absolute_path(str(source))):
            _source = os.listdir(d)
        else:
            _source = [str(source), ]

        return _source

    def postprocess(self, source, result, **kwargs):
        if self.task_cfg.save:
            f_name = os.path.splitext(os.path.basename(source))[0]
            os.makedirs(self.task_cfg.project, exist_ok=True)
            filename = os.path.join(
                self.task_cfg.project,
                f"{f_name}_{self.task_cfg.imgsz}px.jpg")
            result.save(filename)
            logger.info(f"save predicted image to '{filename}'")

    def train(self, *args, **kwargs):
        if self.task_cfg.mode == "train":
            self.task_cfg = self.virtual_cast_train()
        self.trainer.train(self._sub_task_cfg)

    def virtual_cast_predict(self) -> YoloPredictingConfig:
        return self.task_cfg

    def virtual_cast_train(self) -> YoloTrainingConfig:
        return self.task_cfg
