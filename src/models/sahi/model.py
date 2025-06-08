#  Copyright (c) 2025  Van Minh Nguyen
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import os
import logging
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from src.engine.model import Model, ModelConfig
from src.engine.task import TaskConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SahiModel(Model):
    def __init__(
       self,
       cfg: ModelConfig,
       task_cfg: TaskConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.mode_cfg = task_cfg

        logger.info(f"Sahi config: {cfg}")
        logger.info(f"Sahi mode config: {task_cfg}")

        logger.info("construct model")
        self.model = AutoDetectionModel.from_pretrained(
           model_path=self.mode_cfg.model,
           model_type=self.mode_cfg.model_type,
           confidence_threshold=self.mode_cfg.conf_threshold,
           device=self.mode_cfg.device,
           image_size=self.mode_cfg.image_size,
        )

    def predict(self, source, *args, **kwargs):
        result = get_sliced_prediction(
            image=source,
            detection_model=self.model,
            slice_height=self.mode_cfg.slice_height,
            slice_width=self.mode_cfg.slice_width,
            overlap_height_ratio=self.mode_cfg.overlap_height_ratio,
            overlap_width_ratio=self.mode_cfg.overlap_width_ratio,
        )
        if self.mode_cfg.save:
            file_name = os.path.splitext(os.path.basename(source))[0]
            os.makedirs(self.mode_cfg.project, exist_ok=True)
            result.export_visuals(
               export_dir=self.mode_cfg.project,
               hide_conf=self.mode_cfg.hide_conf,
               file_name=file_name,
               rect_th=self.mode_cfg.rect_th,
               text_size=self.mode_cfg.text_size,
               hide_labels=self.mode_cfg.hide_labels,
            )
