#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT License.

import os
from typing import Optional
from src.engine.model import ModelConfig
from src.engine.task import TaskConfig
from dataclasses import dataclass, field


@dataclass
class SahiConfig(ModelConfig):
    ...


@dataclass
class SahiPredictingConfig(TaskConfig):
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "ultralytics/huggingface/torchvision/..."},
    )
    conf_threshold: Optional[float] = field(
        default=0.25,
        metadata={"help": "confidence score threshold"},
    )
    hide_conf: Optional[bool] = field(
        default=True,
        metadata={"help": "hide confidence score or not"},
    )
    slice_height: Optional[int] = field(
        default=256,
        metadata={"help": "heigh of slicer"},
    )
    slice_width: Optional[int] = field(
        default=256,
        metadata={"help": "width of slicer"},
    )
    overlap_height_ratio: Optional[float] = field(
        default=0.2,
        metadata={"help": "overlap height ratio"},
    )
    overlap_width_ratio: Optional[float] = field(
        default=0.2,
        metadata={"help": "overlap width ratio"},
    )
    rect_th: Optional[int] = field(
        default=2,
        metadata={'help': "thickness of rectangular edges"},
    )
    text_size: Optional[float] = field(
        default=0.5,
        metadata={"help": "text size"},
    )
    hide_labels: Optional[bool] = field(
        default=False,
        metadata={"help": "hide labels or not"},
    )
    image_size: Optional[int] = field(
        default=640,
        metadata={"help": "image size"},
    )

    def __post_init__(self):
        if self.model_type is None:
            self.model_type = "ultralytics"
        if self.model_name is None:
            self.model_name = "yolo11n"
        if self.model is None:
            self.model = f"models/pretrained/{self.model_name}.pt"
            assert os.path.isfile(self.model), f"This checkpoint is not on machine '{self.model}"
        if self.project is None:
            self.project = f"assets/test_pretrained_results/sahi/{self.model_name}"