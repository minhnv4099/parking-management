#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license

import os.path
import logging
from typing import Optional
from hydra.utils import get_original_cwd, to_absolute_path
from dataclasses import dataclass, field
from src.engine.task import TaskConfig
from src.engine.model import  ModelConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class SamConfig(ModelConfig):
    ...


@dataclass
class SamPredictingConfig(TaskConfig):
    export_box: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether export box"},
    )
    box_dir: Optional[str] = field(
        default=None,
        metadata={"help": "directory save box files"},
    )
    prompt_type: Optional[str] = field(
        default=None,
        metadata={"help": "type of prompt (e.g. boxes, points)"},
    )
    prompt_dir: Optional[str] = field(
        default=None,
        metadata={"help": "location containing prompt files"},
    )
    show_masks: Optional[bool] = field(
        default=True,
        metadata={"help": "whether show masks"},
    )
    PROMPT_TYPES: Optional[list] = field(
        default_factory=lambda: ["points", ],
        metadata={"help": "types of prompt"}
    )

    line_width: Optional[int] = field(
        default=1,
        metadata={"help": "width of line"},
    )
    show_labels: Optional[bool] = field(
        default=False,
        metadata={"help": "whether show labels"},
    )
    show_boxes: Optional[bool] = field(
        default=True,
        metadata={"help": "whether show boxes"}
    )

    def __post_init__(self):
        if self.model_name is None: self.model_name = "sam2.1_b"
        if self.ckpt is None:
            self.ckpt = f"models/pretrained/{self.model_name}.pt"
            self.abs_ckpt = to_absolute_path(self.ckpt)
            assert os.path.isfile(self.abs_ckpt), f"This checkpoint is not on machine '{self.abs_ckpt}"
        if self.project is None:
            self.project = f"assets/test_pretrained_results/sam/{self.model_name}"
        if get_original_cwd() != os.getcwd():
            self.project = os.path.join(
                os.getcwd(),
                "sam",
                self.model_name
            )
        if self.prompt_type not in self.PROMPT_TYPES:
            self.prompt_type = None
        if self.prompt_type and self.prompt_dir is None:
            self.prompt_dir = to_absolute_path(f"data/external/{self.prompt_type}/txts/")
        if self.export_box and self.box_dir is None:
            self.box_dir = to_absolute_path("data/iterm/boxes/")

        self.EXCLUSIVE_KEYS = [
            "abs_ckpt",
            "export_box",
            "prompt_dir",
            "show_masks",
            "model_name",
            "save_run",
            "ckpt",
            "PROMPT_TYPES",
            "prompt_type",
            "box_dir",
            "EXCLUSIVE_KEYS"
        ]


@dataclass
class SamTrainingConfig(TaskConfig):
    ...
