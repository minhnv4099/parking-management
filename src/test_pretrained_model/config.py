#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#
import os.path
from dataclasses import dataclass, field

from sympy.physics.units import femto


@dataclass
class PretrainedModelConfig:
    ...


@dataclass(init=True)
class YoloConfig(PretrainedModelConfig):
    model_path: str = field(default_factory=lambda: None)
    model_type: str = field(default_factory=lambda: None)
    task: str = field(default="detect")
    mode: str = field(default="predict")
    imgsz_predict: int = field(default=640)
    imgsz_train: int = field(default=640)
    project: str = field(default=None)
    device: str = field(default="cpu")
    save: bool = field(default=False)

    def __post_init__(self):
        if self.model_type is None:
            self.model_type = "yolo11n"
        if self.model_path is None:
            self.model_path = f"models/pretrained/{self.model_type}.pt"
            assert os.path.isfile(self.model_path), "This checkpoint is not on machine"
        if self.project is None:
            self.project = f"assets/test_pretrained_results/standard_yolo/{self.model_type}"


@dataclass(init=True)
class SahiConfig(PretrainedModelConfig):
    model_type: str = field(default_factory=lambda: None)
    model_name: str = field(default_factory=lambda: None)
    model_path: str = field(default_factory=lambda: None)
    conf_threshold: float = field(default=0.25)
    hide_conf: bool = field(default=True)
    slice_height: int = field(default=256)
    slice_width: int = field(default=256)
    overlap_height_ratio: float = field(default=0.2)
    overlap_width_ratio: float = field(default=0.2)
    project: str = field(default=None)
    device: str = field(default="cpu")
    save: bool = field(default=False)
    rect_th: int = field(default=2)
    text_size: float = field(default=0.7)
    hide_labels: bool = field(default=False)

    def __post_init__(self):
        if self.model_type is None:
            self.model_type = "ultralytics"
        if self.model_name is None:
            self.model_name = "yolo11n"
        if self.model_path is None:
            self.model_path = f"models/pretrained/{self.model_name}.pt"
            assert os.path.isfile(self.model_path), "This checkpoint is not on machine"
        if self.project is None:
            self.project = f"assets/test_pretrained_results/sahi/{self.model_name}"
