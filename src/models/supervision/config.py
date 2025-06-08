#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT License.

import os
from typing import Optional
from dataclasses import dataclass, field
from src.configs.config import ModelConfig
from src.tasks.task import TaskConfig
from supervision.draw.color import Color


@dataclass
class SupervisionConfig(ModelConfig):
    ...


@dataclass
class SvPredictingConfig(TaskConfig):
    slice_wh: Optional[tuple] = field(
        default_factory=lambda: (256, 256),
        metadata={"help": "width and height of slicer"},
    )
    overlap_wh: Optional[tuple] = field(
        default_factory=lambda: (0.2, 0.2),
        metadata={"help": "width and heigh overlap"},
    )
    thickness: Optional[int] = field(
        default=2,
        metadata={"help": "thickness of box edges"},
    )
    text_scale: Optional[float] = field(
        default=0.5,
        metadata={"help": "text scale"},
    )
    text_padding: Optional[int] = field(
        default=2,
        metadata={"help": "text padding"},
    )
    text_color: Optional[Color] = field(
        default=Color.WHITE,
        metadata={"help": "text color"},
    )
    box_color: Optional[Color] = field(
        default=Color.BLUE,
        metadata={"help": "box color"}
    )
    label_color: Optional[Color] = field(
        default=Color.BLACK,
        metadata={"help": "label color"}
    )
    task: Optional[str] = field(
        default="detect",
        metadata={"help": "task"},
    )
    text_thickness: Optional[int] = field(
        default=1,
        metadata={"help": "text thickness"},
    )
    border_radius: Optional[int] = field(
        default=0,
        metadata={"help": "border radius"},
    )

    def __post_init__(self):
        if self.task is None:
            self.task = "detect"
        if self.model_name is None:
            self.model_name = 'yolo11n'
        if self.model is None:
            self.model = f"models/pretrained/{self.model_name}.pt"
            assert os.path.isfile(self.model), f"This checkpoint is not on machine '{self.model}"
        if self.project is None:
            self.project = f"assets/test_pretrained_results/supervision/{self.model_name}"
