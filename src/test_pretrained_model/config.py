#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#
import os.path
from dataclasses import dataclass, field
from supervision.draw.color import Color


@dataclass
class PretrainedModelConfig:
    model_name: str = field(default=None)
    model_path: str = field(default=None)
    project: str = field(default=None)
    save: bool = field(default=False)
    device: str = field(default="cpu")
    task: str = field(default=None)
    mode: str = field(default=None)


@dataclass(init=True)
class YoloConfig(PretrainedModelConfig):
    imgsz_predict: int = field(default=640)
    imgsz_train: int = field(default=640)
    save_run: bool = field(default=False)
    model_task: str = field(default='yolo')

    TASK_SUFFIX: dict = field(
        default_factory=lambda: dict(
            detect='',
            seg='-seg',),
        repr=False,
    )

    def __post_init__(self):
        if self.task is None:
            self.task = "detect"
        if self.mode is None:
            self.mode = "predict"
        if self.model_name is None:
            self.model_name = "yolo11n"
        self.model_task = 'yolo' + self.TASK_SUFFIX[self.task]
        if self.model_path is None:
            self.model_path = f"models/pretrained/{self.model_name + self.TASK_SUFFIX[self.task]}.pt"
            assert os.path.isfile(self.model_path), f"This checkpoint is not on machine '{self.model_path}"
        if self.project is None:
            self.project = f"assets/test_pretrained_results/{self.model_task}/{self.model_name}"



@dataclass(init=True)
class SahiConfig(PretrainedModelConfig):
    model_type: str = field(default_factory=lambda: None)
    conf_threshold: float = field(default=0.25)
    hide_conf: bool = field(default=True)
    slice_height: int = field(default=256)
    slice_width: int = field(default=256)
    overlap_height_ratio: float = field(default=0.2)
    overlap_width_ratio: float = field(default=0.2)
    rect_th: int = field(default=2)
    text_size: float = field(default=0.5)
    hide_labels: bool = field(default=False)

    def __post_init__(self):
        if self.model_type is None:
            self.model_type = "ultralytics"
        if self.model_name is None:
            self.model_name = "yolo11n"
        if self.model_path is None:
            self.model_path = f"models/pretrained/{self.model_name}.pt"
            assert os.path.isfile(self.model_path), f"This checkpoint is not on machine '{self.model_path}"
        if self.project is None:
            self.project = f"assets/test_pretrained_results/sahi/{self.model_name}"


@dataclass(init=True)
class SupervisionConfig(PretrainedModelConfig):
    slice_wh: tuple = field(default_factory=lambda: (256, 256))
    overlap_ratio_wh: tuple = field(default_factory=lambda: (0.2, 0.2))
    thickness: int = field(default=2)
    text_scale: float = field(default=0.5)
    text_padding: int = field(default=2)
    text_color: Color = field(default=Color.WHITE)
    color: Color = field(default=Color.ROBOFLOW)

    def __post_init__(self):
        if self.task is None:
            self.task = "detect"
        if self.model_name is None:
            self.model_name = 'yolo11n'
        if self.model_path is None:
            self.model_path = f"models/pretrained/{self.model_name}.pt"
            assert os.path.isfile(self.model_path), f"This checkpoint is not on machine '{self.model_path}"
        if self.project is None:
            self.project = f"assets/test_pretrained_results/supervision/{self.model_name}"


@dataclass(init=True)
class SamConfig(PretrainedModelConfig):
    export_box: bool = field(default=False)
    box_dir: str = field(default=None)
    save_run: bool = field(default=False)
    prompt_type: str = field(default=None)
    prompt_dir: str = field(default=None)
    line_width: int = field(default=1)
    show_labels: bool = field(default=False)
    show_masks: bool = field(default=True)
    show_boxes: bool = field(default=True)
    PROMPT_TYPES = ["points",]

    def __post_init__(self):
        self.task = "seg"
        if self.model_name is None:
            self.model_name = "sam2.1_b"
        if self.model_path is None:
            self.model_path = f"models/pretrained/{self.model_name}.pt"
            assert os.path.isfile(self.model_path), f"This checkpoint is not on machine '{self.model_path}"
        if self.project is None:
            self.project = f"assets/test_pretrained_results/sam/{self.model_name}"
        if self.prompt_type not in self.PROMPT_TYPES:
            self.prompt_type = None
        if self.prompt_type and self.prompt_dir is None:
            self.prompt_dir = f"data/external/{self.prompt_type}/txts/"
        if self.export_box and self.box_dir is None:
            self.box_dir = "data/iterm/boxes/"
