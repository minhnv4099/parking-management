#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license

import os
import cv2
import logging
import supervision as sv
import numpy as np
from ultralytics import YOLO
from src.models.model import BaseModel
from .config import ModelConfig, TaskConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class SupervisionModel(BaseModel):
    def __init__(
        self,
        cfg: ModelConfig = None,
        task_cfg: TaskConfig = None,
    ):
        self.cfg = cfg
        self.task_cfg = task_cfg

        logger.info(f"Supervision config: {cfg}")
        logger.info(f"Supervision task config: {task_cfg}")

        logger.info("construct model")

        self.predictor = YOLO(
            model=self.task_cfg.model,
            task=self.task_cfg.task,
            verbose=True,
        )

        def callback(image_slice: np.ndarray) -> sv.Detections:
            result = self.predictor(image_slice)[0]
            return sv.Detections.from_ultralytics(result)

        self.slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=(256, 256),
            overlap_wh=(50, 50),
            overlap_ratio_wh=None,
        )

    def predict(self, source, *args, **kwargs):
        image = cv2.imread(source)

        detections = self.slicer(image)
        box_annotator = sv.BoxAnnotator(
            thickness=self.task_cfg.thickness,
            color=self.task_cfg.box_color,
        )
        label_annotator = sv.LabelAnnotator(
            text_scale=self.task_cfg.text_scale,
            text_padding=self.task_cfg.text_padding,
            text_color=self.task_cfg.text_color,
            color=self.task_cfg.label_color,
            text_thickness=self.task_cfg.text_thickness,
            border_radius=self.task_cfg.border_radius,
        )

        annotated_image = box_annotator.annotate(
            scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)

        if self.task_cfg.save:
            f_name = os.path.splitext(os.path.basename(source))[0]
            os.makedirs(self.task_cfg.project, exist_ok=True)
            cv2.imwrite(
                os.path.join(self.task_cfg.project, f"{f_name}.jpg"),
                annotated_image,
            )
